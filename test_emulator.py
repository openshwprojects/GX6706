import struct
import unittest
import numpy as np
from gx6706_emu import (
    GX6706Emulator,
    STOP_BKPT,
    STOP_STAGE2_ENTRY,
    STOP_STAGE3_ENTRY,
    ST_PC,
    ST_INSN_COUNT,
)


class TestCSKYv1ISA(unittest.TestCase):
    def setUp(self):
        self.emu = GX6706Emulator()

    def _run_code(self, insns, start_pc=0x90000000, max_steps=100):
        off = start_pc - 0x90000000
        for i, insn in enumerate(insns):
            self.emu.dram_u16[(off >> 1) + i] = insn
        # Append bkpt (0x0000)
        self.emu.dram_u16[(off >> 1) + len(insns)] = 0x0000
        self.emu.state[ST_PC] = start_pc
        return self.emu.step(max_steps=max_steps)

    def test_xtrb_ff1_brev(self):
        self.emu.regs[2] = 0x12345678
        self._run_code([
            0x0132,  # xtrb0 r1, r2 -> 0x12
            0x1213,  # mov r3, r1
            0x0122,  # xtrb1 r1, r2 -> 0x34
            0x1214,  # mov r4, r1
            0x0112,  # xtrb2 r1, r2 -> 0x56
            0x1215,  # mov r5, r1
            0x0102,  # xtrb3 r1, r2 -> 0x78
            0x1216,  # mov r6, r1
        ])
        self.assertEqual(self.emu.regs[3], 0x12)
        self.assertEqual(self.emu.regs[4], 0x34)
        self.assertEqual(self.emu.regs[5], 0x56)
        self.assertEqual(self.emu.regs[6], 0x78)

        # ff1 and brev
        self.emu.regs[2] = 0x00800000
        self.emu.regs[3] = 0x00000000
        self.emu.regs[4] = 0x80000001
        self._run_code([
            0x00e2,  # ff1 r2 -> 8
            0x00e3,  # ff1 r3 -> 32
            0x00f4,  # brev r4 -> 0x80000001
        ])
        self.assertEqual(self.emu.regs[2], 8)
        self.assertEqual(self.emu.regs[3], 32)
        self.assertEqual(self.emu.regs[4], 0x80000001)

    def test_64bit_addc_subc(self):
        # 64-bit addition: (0x00000001_ffffffff) + (0x00000002_00000002) = 0x00000004_00000001
        self.emu.regs[2] = 0xffffffff
        self.emu.regs[3] = 0x00000001
        self.emu.regs[4] = 0x00000002
        self.emu.regs[5] = 0x00000002
        self._run_code([
            0x0d22,  # cmplt r2, r2 (clrc: C=0)
            0x0642,  # addc r2, r4
            0x0653,  # addc r3, r5
        ])
        self.assertEqual(self.emu.regs[2], 0x00000001)
        self.assertEqual(self.emu.regs[3], 0x00000004)

        # 64-bit subtraction: (0x00000004_00000001) - (0x00000001_ffffffff) = 0x00000002_00000002
        self.emu.regs[6] = 0xffffffff
        self.emu.regs[7] = 0x00000001
        self._run_code([
            0x0c22,  # cmphs r2, r2 (setc: C=1, no borrow)
            0x0762,  # subc r2, r6
            0x0773,  # subc r3, r7
        ])
        self.assertEqual(self.emu.regs[2], 0x00000002)
        self.assertEqual(self.emu.regs[3], 0x00000002)

    def test_stm_ldm_stq_ldq(self):
        self.emu.regs[0] = 0x90100000
        for i in range(4, 16):
            self.emu.regs[i] = 0x1000 + i
        self._run_code([
            0x0078,  # stm r8-r15, (r0)
            0x0050,  # stq r4-r7, (r0) -> overwrites first 16 bytes at r0
        ])
        # Verify stq wrote r4..r7 at 0x90100000..0x9010000c and stm wrote r12..r15 at +16..+28
        for i in range(4, 8):
            self.emu.regs[i] = 0
        self._run_code([
            0x0040,  # ldq r4-r7, (r0)
        ])
        for i in range(4, 8):
            self.assertEqual(self.emu.regs[i], 0x1000 + i)

    def test_stage1_boot_to_stage2(self):
        self.emu.load_stage1()
        reason = self.emu.step(max_steps=5_000_000, stop_on_stage_transition=True)
        uart_out = self.emu.get_uart_output()
        print(f"[Stage 1] stop_reason={reason}, insns={self.emu.state[ST_INSN_COUNT]}, pc=0x{self.emu.state[ST_PC]:08x}, uart={uart_out!r}")
        self.assertEqual(reason, STOP_STAGE2_ENTRY)
        self.assertEqual(self.emu.state[ST_PC], 0x9316b400)
        self.assertEqual(uart_out, "RUkgd:0\r\n5N")
        # Verify Stage 2 in DRAM at 0x9316b400 matches flash[0x2020:0x2020+0x1aef0]
        dram_s2 = bytes(self.emu.dram[0x0316b400:0x0316b400 + 0x1aef0])
        flash_s2 = bytes(self.emu.flash[0x2020:0x2020 + 0x1aef0])
        self.assertEqual(dram_s2, flash_s2)

        # Now continue into Stage 2 (GxLoader) and verify it reaches Stage 3 (0x90000000)
        reason2 = self.emu.step(max_steps=50_000_000, stop_on_stage_transition=True)
        uart_out2 = self.emu.get_uart_output()
        print(f"\n[Stage 2] stop_reason={reason2}, insns={self.emu.state[ST_INSN_COUNT]}, pc=0x{self.emu.state[ST_PC]:08x}")
        self.assertEqual(reason2, STOP_STAGE3_ENTRY)
        self.assertEqual(self.emu.state[ST_PC], 0x90000000)
        self.assertIn("GxLoader", uart_out2)
        self.assertIn("Decompress kernel ...ok", uart_out2)

        atag_bytes = bytes(self.emu.dram[0x03186318:0x03186480])
        print(f"[Stage 2 -> 3 ATAG @ 0x93186318]: {atag_bytes[:128]!r}")

        # Now continue into Stage 3 (eCos + DVB-T2 application) until OSD UI is rendered
        reason3 = self.emu.step(max_steps=135_000_000, stop_on_stage_transition=False)
        uart_out3 = self.emu.get_uart_output()
        print(f"\n[Stage 3] stop_reason={reason3}, insns={self.emu.state[ST_INSN_COUNT]}, pc=0x{self.emu.state[ST_PC]:08x}")
        print("=== UART OUTPUT AFTER STAGE 3 UI BOOT ===")
        print(uart_out3)
        print("=========================================")

        self.assertIn("memptr : 0x9065a000", uart_out3)
        self.assertIn("now,2024-01-15 Tuesday 22:47:11", uart_out3)
        self.assertIn("E/app_av_get_cvbs_standard:676", uart_out3)

        # Verify OSD region descriptor & rendered framebuffer
        osd_info = self.emu.get_osd_region_info()
        self.assertTrue(osd_info["enabled"])
        self.assertEqual(osd_info["head_phys"], 0x106d2000)
        self.assertEqual(osd_info["width"], 1024)
        self.assertEqual(osd_info["height"], 576)
        self.assertEqual(osd_info["fb_phys"], 0x11f950b8)

        fb_off = (osd_info["fb_phys"] & 0x03ffffff) >> 1
        fb16 = self.emu.dram_u16[fb_off : fb_off + 1024 * 576]
        ui_pixels = int(np.count_nonzero((fb16 != 0) & (fb16 != 0x8df7)))
        unique_colors = len(np.unique(fb16))
        print(f"[OSD] {osd_info['width']}x{osd_info['height']} @ 0x{osd_info['fb_phys']:08x}: ui_pixels={ui_pixels}, unique_colors={unique_colors}")
        self.assertEqual(ui_pixels, 174720)
        self.assertEqual(unique_colors, 139)

        rgb = self.emu.render_osd(output_path="osd_output.png")
        self.assertEqual(rgb.shape, (576, 1024, 3))

    def test_ga_fillrect_and_blit_colorkey(self):
        # Set up 2 GA command descriptors at 0x10800000:
        # Cmd 0: FillRect 4x4 at dst=0x11000000 (pitch=4) with RGB565 0x1234
        c0 = 0x00800000 >> 2
        self.emu.dram_u32[c0 + 0] = 0x10000404
        self.emu.dram_u32[c0 + 3] = 0xde54c104
        self.emu.dram_u32[c0 + 8] = 0x1234
        self.emu.dram_u32[c0 + 14] = 0x11000000
        self.emu.dram_u32[c0 + 18] = (4 << 18) | (4 << 4) | 1
        self.emu.dram_u32[c0 + 19] = (4 << 18)

        # Prepare 2x2 source at 0x11001000 with colorkey 0xf81f in 2 corners and 0x5678 in 2 corners
        s_off = 0x01001000 >> 1
        self.emu.dram_u16[s_off + 0] = 0x5678
        self.emu.dram_u16[s_off + 1] = 0xf81f
        self.emu.dram_u16[s_off + 2] = 0xf81f
        self.emu.dram_u16[s_off + 3] = 0x9abc

        # Cmd 1: Blit 2x2 from 0x11001000 to 0x11000000 (pitch=4) with colorkey=0xf81f
        c1 = (0x00800000 + 80) >> 2
        self.emu.dram_u32[c1 + 0] = 0x80100404
        self.emu.dram_u32[c1 + 3] = 0xde54c104
        self.emu.dram_u32[c1 + 8] = 0xf81f
        self.emu.dram_u32[c1 + 12] = 0x11001000
        self.emu.dram_u32[c1 + 14] = 0x11000000
        self.emu.dram_u32[c1 + 15] = (2 << 18) | (2 << 4)
        self.emu.dram_u32[c1 + 16] = (2 << 18)
        self.emu.dram_u32[c1 + 18] = (4 << 18) | (2 << 4) | 1
        self.emu.dram_u32[c1 + 19] = (2 << 18)

        # Trigger GA batch of 2 commands via MMIO writes from CPU
        self.emu.regs[2] = 0xa4700000
        self.emu.regs[3] = 0x10800000
        self.emu.regs[4] = 160
        self.emu.regs[5] = 0x00010002
        self._run_code([
            0x9312,  # st.w r3, (r2, 4)  -> GA_CMD_ADDR = 0x10800000
            0x9422,  # st.w r4, (r2, 8)  -> GA_CMD_LEN  = 160
            0x9562,  # st.w r5, (r2, 24) -> GA_CMD_START = 0x10002
        ])

        d_off = 0x01000000 >> 1
        self.assertEqual(self.emu.dram_u16[d_off + 0], 0x5678)
        self.assertEqual(self.emu.dram_u16[d_off + 1], 0x1234)  # colorkey skipped!
        self.assertEqual(self.emu.dram_u16[d_off + 4], 0x1234)  # colorkey skipped!
        self.assertEqual(self.emu.dram_u16[d_off + 5], 0x9abc)

    def test_remote_and_panel_input_screen_navigation(self):
        self.emu.boot_to_gui()

        # 1. Verify initial Installation Guide screen
        self.assertEqual(self.emu.get_active_window(), "list_installation_guide")
        self.assertIn("win_t2_installation_guide", self.emu.get_focus_path())
        flips_ig = self.emu.get_vpu_flip_count()
        rgb_ig0 = self.emu.render_osd(transparent_to_black=True).copy()

        # 2. Navigate Installation Guide with IR remote UP -> focuses Channel Search row
        self.emu.press_key("UP")
        self.assertGreater(self.emu.get_vpu_flip_count(), flips_ig)
        rgb_ig1 = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_ig0, rgb_ig1))

        # 3. Press EXIT on Installation Guide -> opens Main Menu on Program tab
        self.emu.press_key("EXIT")
        self.assertEqual(self.emu.get_active_window(), "btn_main_menu_item_focus")
        self.assertIn("wnd_main_menu_item", self.emu.get_focus_path())
        self.assertEqual(self.emu.get_main_menu_tab(), "Program")

        # 4. Enter Program sub-items (OK) and trigger 'No TV channel!' modal popup (OK)
        self.emu.press_key("OK")
        self.assertEqual(self.emu.get_active_window(), "wnd_main_menu_item_listview")
        self.emu.press_key("OK")
        self.assertEqual(self.emu.get_active_window(), "wnd_pop_tip")

        # 5. Dismiss modal popup via front-panel OK button, move down with CH-, exit sub-list with EXIT
        self.emu.press_panel_button("OK")
        self.assertEqual(self.emu.get_active_window(), "wnd_main_menu_item_listview")
        self.emu.press_panel_button("CH-")
        self.assertEqual(self.emu.get_active_window(), "wnd_main_menu_item_listview")
        self.emu.press_panel_button("EXIT")
        self.assertEqual(self.emu.get_active_window(), "btn_main_menu_item_focus")

        # 6. Navigate across all 7 Main Menu tabs and enter deeper sub-dialogs on each tab
        # Picture tab -> OSD Settings (Row 4 -> wnd_system_setting)
        self.emu.press_panel_button("PANEL_VOL+")
        self.assertEqual(self.emu.get_main_menu_tab(), "Picture")
        rgb_pic = self.emu.render_osd(transparent_to_black=True).copy()
        self.emu.send_keys(["OK", "DOWN", "DOWN", "DOWN", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "cmb_system_setting_opt1")
        self.assertIn("wnd_system_setting", self.emu.get_focus_path())
        rgb_osd_set = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_pic, rgb_osd_set))
        self.emu.send_keys(["EXIT", "EXIT"])
        self.assertEqual(self.emu.get_active_window(), "btn_main_menu_item_focus")

        # Search tab -> Manual Search (Row 3 -> wnd_common_search_setting)
        self.emu.press_panel_button("PANEL_RIGHT")
        self.assertEqual(self.emu.get_main_menu_tab(), "Search")
        self.emu.send_keys(["OK", "DOWN", "DOWN", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "box_cst_options")
        self.assertIn("wnd_common_search_setting", self.emu.get_focus_path())
        rgb_man_search = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_osd_set, rgb_man_search))
        self.emu.send_keys(["EXIT", "EXIT"])

        # Time tab -> Timer Setting (Row 2 -> wnd_timer_setting)
        self.emu.press_key("RIGHT")
        self.assertEqual(self.emu.get_main_menu_tab(), "Time")
        self.emu.send_keys(["OK", "DOWN", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "list_timer_setting_timer")
        self.assertIn("wnd_timer_setting", self.emu.get_focus_path())
        rgb_timer = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_man_search, rgb_timer))
        self.emu.send_keys(["EXIT", "EXIT"])

        # Option tab
        self.emu.press_key("RIGHT")
        self.assertEqual(self.emu.get_main_menu_tab(), "Option")

        # System tab -> Factory Reset password dialog, System Information, Software Upgrade, Network Setting
        self.emu.press_key("RIGHT")
        self.assertEqual(self.emu.get_main_menu_tab(), "System")
        self.emu.send_keys(["OK", "DOWN", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "btn_passwd_input_1")
        self.assertIn("wnd_passwd", self.emu.get_focus_path())
        rgb_passwd = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_timer, rgb_passwd))

        self.emu.send_keys(["EXIT", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "box_system_info_para")
        self.assertIn("wnd_system_information", self.emu.get_focus_path())
        rgb_sysinfo = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_passwd, rgb_sysinfo))

        self.emu.send_keys(["EXIT", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "wnd_common_2nd_menu_listview")
        self.assertIn("wnd_common_2nd_menu", self.emu.get_focus_path())
        rgb_swupg = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_sysinfo, rgb_swupg))

        self.emu.send_keys(["EXIT", "DOWN", "OK"])
        self.assertEqual(self.emu.get_active_window(), "box_network_options")
        self.assertIn("wnd_network", self.emu.get_focus_path())
        rgb_net = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_swupg, rgb_net))
        self.emu.send_keys(["EXIT", "EXIT"])

        # USB tab -> Media Centre (wnd_common_2nd_menu) and wrap back to Program tab
        self.emu.press_key("RIGHT")
        self.assertEqual(self.emu.get_main_menu_tab(), "USB")
        self.emu.send_keys(["OK", "OK"])
        self.assertEqual(self.emu.get_active_window(), "wnd_common_2nd_menu_listview")
        self.assertIn("wnd_common_2nd_menu", self.emu.get_focus_path())
        rgb_usb = self.emu.render_osd(transparent_to_black=True).copy()
        self.assertFalse(np.array_equal(rgb_net, rgb_usb))
        self.emu.send_keys(["EXIT", "EXIT", "RIGHT"])
        self.assertEqual(self.emu.get_main_menu_tab(), "Program")

        # 7. Return to Installation Guide (MENU) and launch DVB T2 Search (UP, OK)
        self.emu.press_key("MENU")
        self.assertEqual(self.emu.get_active_window(), "list_installation_guide")
        self.emu.send_keys(["UP", "OK"])
        self.assertEqual(self.emu.get_active_window(), "btn_search_exit")
        self.assertIn("wnd_search", self.emu.get_focus_path())


if __name__ == "__main__":
    unittest.main()

