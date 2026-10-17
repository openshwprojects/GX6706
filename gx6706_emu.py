import os
import struct
import lzma
import time
import numpy as np
from numba import njit

# Stop reasons
STOP_RUNNING = 0
STOP_MAX_STEPS = 1
STOP_BKPT = 2
STOP_STAGE2_ENTRY = 3
STOP_STAGE3_ENTRY = 4
STOP_UNHANDLED_INSN = 5
STOP_PC_OOB = 6
STOP_INFINITE_LOOP = 7
STOP_LZMA_DECODE = 8

# State array indices
ST_PC = 0
ST_HI = 1
ST_LO = 2
ST_INSN_COUNT = 3
ST_STOP_REASON = 4
ST_SPI_STATE = 5
ST_SPI_CMD = 6
ST_SPI_ADDR = 7
ST_SPI_RX_AVAIL = 8
ST_SPI_JEDEC_IDX = 9
ST_UART_TX_LEN = 10
ST_MMIO_LOG_LEN = 11
ST_IRQ_RAW_LO = 12
ST_IRQ_RAW_HI = 13
ST_SPI_JUST_DATA = 14
ST_LAST_PC = 15
ST_KEY_CODE = 16
ST_IG_SYNCED = 17


@njit( cache=True )
def _brev32(x_in):
    x = np.uint64(x_in)
    x = ((x >> np.uint64(1)) & np.uint64(0x55555555)) | ((x & np.uint64(0x55555555)) << np.uint64(1))
    x = ((x >> np.uint64(2)) & np.uint64(0x33333333)) | ((x & np.uint64(0x33333333)) << np.uint64(2))
    x = ((x >> np.uint64(4)) & np.uint64(0x0f0f0f0f)) | ((x & np.uint64(0x0f0f0f0f)) << np.uint64(4))
    x = ((x >> np.uint64(8)) & np.uint64(0x00ff00ff)) | ((x & np.uint64(0x00ff00ff)) << np.uint64(8))
    x = ((x >> np.uint64(16)) & np.uint64(0x0000ffff)) | ((x & np.uint64(0x0000ffff)) << np.uint64(16))
    return np.uint32(x)


@njit( cache=True )
def _ff1_32(x_in):
    x = np.uint64(x_in)
    if x == 0:
        return np.uint32(32)
    n = np.uint32(0)
    if (x & np.uint64(0xffff0000)) == 0:
        n += np.uint32(16)
        x <<= np.uint64(16)
    if (x & np.uint64(0xff000000)) == 0:
        n += np.uint32(8)
        x <<= np.uint64(8)
    if (x & np.uint64(0xf0000000)) == 0:
        n += np.uint32(4)
        x <<= np.uint64(4)
    if (x & np.uint64(0xc0000000)) == 0:
        n += np.uint32(2)
        x <<= np.uint64(2)
    if (x & np.uint64(0x80000000)) == 0:
        n += np.uint32(1)
    return n


@njit( cache=True )
def _spi_tx_byte(b_in, state):
    b = np.uint64(b_in) & np.uint64(0xff)
    st = state[ST_SPI_STATE]
    if st == 0:
        state[ST_SPI_CMD] = b
        if b == 0x03 or b == 0x0b or b == 0x3b or b == 0x6b:
            state[ST_SPI_STATE] = 1
            state[ST_SPI_ADDR] = 0
            state[ST_SPI_JUST_DATA] = 0
        elif b == 0x9f:
            state[ST_SPI_STATE] = 6
            state[ST_SPI_JEDEC_IDX] = 0
            state[ST_SPI_JUST_DATA] = 1
        elif b == 0x05 or b == 0x35 or b == 0x15:
            state[ST_SPI_STATE] = 7
            state[ST_SPI_JUST_DATA] = 1
        else:
            state[ST_SPI_STATE] = 8
            state[ST_SPI_JUST_DATA] = 0
    elif st == 1:
        state[ST_SPI_ADDR] = b << np.uint64(16)
        state[ST_SPI_STATE] = 2
    elif st == 2:
        state[ST_SPI_ADDR] |= b << np.uint64(8)
        state[ST_SPI_STATE] = 3
    elif st == 3:
        state[ST_SPI_ADDR] |= b
        cmd = state[ST_SPI_CMD]
        if cmd == 0x0b or cmd == 0x3b or cmd == 0x6b:
            state[ST_SPI_STATE] = 4
        else:
            state[ST_SPI_STATE] = 5
            state[ST_SPI_JUST_DATA] = 1
    elif st == 4:
        # Dummy byte after 24-bit address for Fast Read (0x0b / 0x3b)
        state[ST_SPI_STATE] = 5
        state[ST_SPI_JUST_DATA] = 1


@njit( cache=True )
def _spi_rx_byte(state, flash):
    st = state[ST_SPI_STATE]
    if st == 5:
        addr = state[ST_SPI_ADDR] & np.uint64(0x3fffff)
        val = np.uint32(flash[addr])
        state[ST_SPI_ADDR] = (addr + np.uint64(1)) & np.uint64(0x3fffff)
        return val
    elif st == 6:
        idx = state[ST_SPI_JEDEC_IDX]
        state[ST_SPI_JEDEC_IDX] = idx + np.uint64(1)
        # Winbond W25Q32 (4MB): 0xef, 0x40, 0x16
        if idx == 0:
            return np.uint32(0xef)
        elif idx == 1:
            return np.uint32(0x40)
        elif idx == 2:
            return np.uint32(0x16)
        return np.uint32(0x00)
    elif st == 7:
        # Status register: WIP=0, WEL=0
        return np.uint32(0x00)
    return np.uint32(0xff)


@njit( cache=True )
def _mmio_read32(phys, state, mmio_low_u32, mmio_vpu_u32, flash):
    if 0x00000000 <= phys < 0x01000000:
        idx = phys >> 2
        # UART0 (0x00402000) & UART1 (0x00403000)
        if (phys & 0xffffe000) == 0x00402000:
            reg = phys & 0xfff
            if reg == 0x14:
                return np.uint32(0x60)  # LSR: THRE | TEMT
            if reg == 0x7c:
                return np.uint32(0x06)  # USR: TFNF | TFE
            if reg == 0x08:
                return np.uint32(0x01)  # IIR: no interrupt pending
        # I2C Controllers (0x00203000, 0x00205000, 0x00206000)
        elif phys == 0x00203000 or phys == 0x00205000 or phys == 0x00206000:
            return np.uint32(0x08)  # I2C transfer complete (bit 3=1) + ACK (bit 1=0)
        # Timers (0x0020a000)
        elif (phys & 0xfffff000) == 0x0020a000:
            reg = phys & 0xfff
            if reg == 0x04 or reg == 0x44 or reg == 0x84:
                # Counter VALUE: advance on read so busy-wait delay loops finish quickly
                c_base = (phys & 0xffffffc0) >> 2
                old = mmio_low_u32[idx]
                cur = np.uint32(old + np.uint32(25))
                if cur < old:
                    cur = mmio_low_u32[c_base + (0x28 >> 2)]
                    cfg = mmio_low_u32[c_base + (0x20 >> 2)]
                    if (cfg & 2) != 0:
                        mmio_low_u32[c_base] = np.uint32(1)
                        c_idx = (reg >> 6) & 3
                        b_addr = 0x00500060 + 10 + c_idx
                        ch = (mmio_low_u32[b_addr >> 2] >> ((b_addr & 3) << 3)) & np.uint32(0x3f)
                        if ch < 32:
                            state[ST_IRQ_RAW_LO] |= np.uint64(1) << np.uint64(ch)
                        else:
                            state[ST_IRQ_RAW_HI] |= np.uint64(1) << np.uint64(ch - 32)
                mmio_low_u32[idx] = cur
                return cur
        # Interrupt Controller (0x00500000)
        elif (phys & 0xfffff000) == 0x00500000:
            reg = phys & 0xfff
            if reg == 0x00:
                nmask_lo = mmio_low_u32[(0x00500050) >> 2]
                return np.uint32(np.uint32(state[ST_IRQ_RAW_LO]) & (~nmask_lo))
            elif reg == 0x04:
                nmask_hi = mmio_low_u32[(0x00500054) >> 2]
                return np.uint32(np.uint32(state[ST_IRQ_RAW_HI]) & (~nmask_hi))
        # DDR Controller (0x00c00000)
        elif phys == 0x00c000b8:
            return np.uint32(0x20)
        # USB OHCI0 (0x00900000) & OHCI1 (0x00901000)
        elif (phys & 0xffffe000) == 0x00900000:
            reg = phys & 0xfff
            if reg == 0x08:
                return np.uint32(mmio_low_u32[idx] & np.uint32(0xfffffffe))
        # USB EHCI (0x00904000)
        elif (phys & 0xfffff000) == 0x00904000:
            reg = phys & 0xfff
            if reg == 0x10:
                return np.uint32(mmio_low_u32[idx] & np.uint32(0xfffffffd))
            elif reg == 0x14:
                usbcmd = mmio_low_u32[(0x00904010) >> 2]
                sts = mmio_low_u32[idx] & np.uint32(0xffff2fff)
                if (usbcmd & 1) == 0:
                    sts |= np.uint32(0x1000)  # STS_HALT
                if (usbcmd & 0x10) != 0:
                    sts |= np.uint32(0x4000)  # STS_PSS
                if (usbcmd & 0x20) != 0:
                    sts |= np.uint32(0x8000)  # STS_ASS
                return np.uint32(sts)
        # NationalChip SPI Master (0x00302000)
        elif (phys & 0xfffff000) == 0x00302000:
            reg = phys & 0xfff
            if reg == 0x04:
                return np.uint32(1)  # SPI_SR: transfer complete
            elif reg == 0x0c:
                cr = mmio_low_u32[(0x00302000) >> 2]
                sz = (cr >> 14) & 3
                if sz == 3:
                    b0 = _spi_rx_byte(state, flash)
                    b1 = _spi_rx_byte(state, flash)
                    b2 = _spi_rx_byte(state, flash)
                    b3 = _spi_rx_byte(state, flash)
                    return np.uint32((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
                elif sz == 2:
                    b0 = _spi_rx_byte(state, flash)
                    b1 = _spi_rx_byte(state, flash)
                    b2 = _spi_rx_byte(state, flash)
                    return np.uint32((b0 << 16) | (b1 << 8) | b2)
                elif sz == 1:
                    b0 = _spi_rx_byte(state, flash)
                    b1 = _spi_rx_byte(state, flash)
                    return np.uint32((b0 << 8) | b1)
                else:
                    return _spi_rx_byte(state, flash)
        # DesignWare SPI (0x00e00000)
        elif (phys & 0xfffff000) == 0x00e00000:
            reg = phys & 0xfff
            if reg == 0x20:
                return np.uint32(0)  # TXFLR
            elif reg == 0x24:
                avail = state[ST_SPI_RX_AVAIL]
                return np.uint32(32 if avail > 32 else avail)
            elif reg == 0x28:
                sr = np.uint32(0x06)
                if state[ST_SPI_RX_AVAIL] > 0:
                    sr = np.uint32(0x0e)
                return sr
            elif reg == 0x5c:
                return np.uint32(0x3130322a)  # "102*"
            elif reg == 0x60:
                ctrlr0 = mmio_low_u32[(0x00e00000) >> 2]
                dfs = ctrlr0 & 0x1f
                tmod = (ctrlr0 >> 10) & 3
                if tmod == 0:
                    tmod = (ctrlr0 >> 8) & 3
                if state[ST_SPI_RX_AVAIL] > 0:
                    state[ST_SPI_RX_AVAIL] -= np.uint64(1)
                if tmod == 0 and state[ST_SPI_JUST_DATA] != 0:
                    state[ST_SPI_JUST_DATA] = 0
                    return np.uint32(0xff)
                if dfs == 31:
                    b0 = _spi_rx_byte(state, flash)
                    b1 = _spi_rx_byte(state, flash)
                    b2 = _spi_rx_byte(state, flash)
                    b3 = _spi_rx_byte(state, flash)
                    return np.uint32((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
                elif dfs == 15:
                    b0 = _spi_rx_byte(state, flash)
                    b1 = _spi_rx_byte(state, flash)
                    return np.uint32((b0 << 8) | b1)
                else:
                    return _spi_rx_byte(state, flash)
        return mmio_low_u32[idx]
    elif 0x04000000 <= phys < 0x05000000:
        vpu_idx = (phys - 0x04000000) >> 2
        if phys == 0x04800014:
            cur = mmio_vpu_u32[vpu_idx]
            line = ((cur >> np.uint32(16)) & np.uint32(0x7ff)) + np.uint32(37)
            field = (cur >> np.uint32(31)) & np.uint32(1)
            if line >= np.uint32(576):
                line -= np.uint32(576)
                field ^= np.uint32(1)
            cur = (field << np.uint32(31)) | (line << np.uint32(16))
            mmio_vpu_u32[vpu_idx] = cur
            return cur
        return mmio_vpu_u32[vpu_idx]
    return np.uint32(0)


@njit( cache=True )
def _ga_exec_cmds(val, state, dram, dram_u32, mmio_vpu_u32, mmio_log):
    prev = mmio_vpu_u32[(0x00700018) >> 2]
    if (val & np.uint32(0x10000)) != 0 and (prev & np.uint32(0x10000)) == 0:
        cmd_cnt = val & np.uint32(0xffff)
        base_phys = mmio_vpu_u32[(0x00700004) >> 2]
        buf_sz = mmio_vpu_u32[(0x00700008) >> 2]
        cur_phys = mmio_vpu_u32[(0x0070000c) >> 2]
        if cur_phys == np.uint32(0):
            cur_phys = base_phys
        off = cur_phys - base_phys
        for _ in range(int(cmd_cnt)):
            c_phys = base_phys + off
            c_dram_w = ((c_phys & np.uint32(0x03ffffff)) >> 2)
            w0 = dram_u32[c_dram_w + 0]
            w3 = dram_u32[c_dram_w + 3]
            w8 = dram_u32[c_dram_w + 8]
            w12 = dram_u32[c_dram_w + 12]
            w14 = dram_u32[c_dram_w + 14]
            w15 = dram_u32[c_dram_w + 15]
            w16 = dram_u32[c_dram_w + 16]
            w18 = dram_u32[c_dram_w + 18]
            w19 = dram_u32[c_dram_w + 19]

            dst_base = int(w14 & np.uint32(0x03ffffff))
            dst_pitch = int(w18 >> np.uint32(18))
            dst_w = int((w18 >> np.uint32(4)) & np.uint32(0x3fff))
            dst_h = int(w19 >> np.uint32(18))
            dst_fmt = int(w3 & np.uint32(0x3f))

            if (w0 & np.uint32(0x80000000)) == 0 and (w0 & np.uint32(0x10000000)) != 0:
                # FillRect
                color = w8
                c_lo = np.uint8(color & np.uint32(0xff))
                c_hi = np.uint8((color >> np.uint32(8)) & np.uint32(0xff))
                for y in range(dst_h):
                    row_off = dst_base + (y * dst_pitch) * (4 if dst_fmt == 7 else 2)
                    if dst_fmt == 7:
                        for x in range(dst_w):
                            dram_u32[(row_off >> 2) + x] = color
                    else:
                        for x in range(dst_w):
                            p = row_off + (x << 1)
                            dram[p] = c_lo
                            dram[p + 1] = c_hi
            elif (w0 & np.uint32(0x80000000)) != 0 and dst_w > 0 and dst_h > 0:
                # Blit / Copy / Zoom
                src_base = int(w12 & np.uint32(0x03ffffff))
                src_pitch = int(w15 >> np.uint32(18))
                src_w = int((w15 >> np.uint32(4)) & np.uint32(0x3fff))
                src_h = int(w16 >> np.uint32(18))
                src_fmt = int((w0 >> np.uint32(8)) & np.uint32(0x3f))
                ckey_en = (w0 & np.uint32(0x00100000)) != 0
                ckey = w8 & np.uint32(0xffff)
                if src_w > 0 and src_h > 0:
                    for y in range(dst_h):
                        sy = (y * src_h) // dst_h
                        s_row = src_base + (sy * src_pitch) * (4 if src_fmt == 7 else 2)
                        d_row = dst_base + (y * dst_pitch) * (4 if dst_fmt == 7 else 2)
                        for x in range(dst_w):
                            sx = (x * src_w) // dst_w
                            if src_fmt == 7:
                                pix = dram_u32[(s_row >> 2) + sx]
                            else:
                                sp = s_row + (sx << 1)
                                pix = np.uint32(dram[sp]) | (np.uint32(dram[sp + 1]) << np.uint32(8))
                            if ckey_en and (pix & np.uint32(0xffff)) == ckey:
                                continue
                            if dst_fmt == 7:
                                dram_u32[(d_row >> 2) + x] = pix
                            else:
                                dp = d_row + (x << 1)
                                dram[dp] = np.uint8(pix & np.uint32(0xff))
                                dram[dp + 1] = np.uint8((pix >> np.uint32(8)) & np.uint32(0xff))

            off += np.uint32(80)
            if buf_sz > np.uint32(0) and off >= buf_sz:
                off = off % buf_sz


@njit( cache=True )
def _mmio_write32(phys, val_in, state, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log):
    val = np.uint32(val_in)
    if (
        (phys & 0xfffff000) != 0x00e00000
        and (phys & 0xfffff000) != 0x00302000
        and (phys & 0xffffe000) != 0x00402000
        and (phys & 0xfffff000) != 0x0020a000
        and (phys & 0xfffff000) != 0x00500000
    ):
        log_idx = state[ST_MMIO_LOG_LEN]
        if log_idx < np.uint64(len(mmio_log)):
            mmio_log[log_idx, 0] = np.uint32(state[ST_PC])
            mmio_log[log_idx, 1] = np.uint32(phys)
            mmio_log[log_idx, 2] = val
            state[ST_MMIO_LOG_LEN] = log_idx + np.uint64(1)

    if 0x00000000 <= phys < 0x01000000:
        idx = phys >> 2
        # UART0 / UART1
        if (phys & 0xffffe000) == 0x00402000:
            reg = phys & 0xfff
            base_idx = (phys & 0xfffff000) >> 2
            if reg == 0x00:
                lcr = mmio_low_u32[base_idx + (0x0c >> 2)]
                if (lcr & 0x80) == 0:
                    ulen = state[ST_UART_TX_LEN]
                    if ulen < np.uint64(len(uart_buf)):
                        uart_buf[ulen] = np.uint8(val & 0xff)
                        state[ST_UART_TX_LEN] = ulen + np.uint64(1)
                    return
        # NationalChip SPI Master (0x00302000)
        elif (phys & 0xfffff000) == 0x00302000:
            reg = phys & 0xfff
            if reg == 0x00:
                mmio_low_u32[idx] = val
                if (val & 0x00030000) == 0:
                    state[ST_SPI_STATE] = 0
                    state[ST_SPI_RX_AVAIL] = 0
                    state[ST_SPI_JUST_DATA] = 0
                return
            elif reg == 0x08:
                mmio_low_u32[idx] = val
                cr = mmio_low_u32[(0x00302000) >> 2]
                sz = (cr >> 14) & 3
                if sz == 3:
                    _spi_tx_byte((val >> 24) & 0xff, state)
                    _spi_tx_byte((val >> 16) & 0xff, state)
                    _spi_tx_byte((val >> 8) & 0xff, state)
                    _spi_tx_byte(val & 0xff, state)
                elif sz == 2:
                    _spi_tx_byte((val >> 16) & 0xff, state)
                    _spi_tx_byte((val >> 8) & 0xff, state)
                    _spi_tx_byte(val & 0xff, state)
                elif sz == 1:
                    _spi_tx_byte((val >> 8) & 0xff, state)
                    _spi_tx_byte(val & 0xff, state)
                else:
                    _spi_tx_byte(val & 0xff, state)
                return
        # System Control (0x0030a000)
        elif (phys & 0xfffff000) == 0x0030a000:
            reg = phys & 0xfff
            if reg == 0x904:
                mmio_low_u32[idx] = val
                state[ST_SPI_STATE] = 0
                state[ST_SPI_RX_AVAIL] = 0
                state[ST_SPI_JUST_DATA] = 0
                return
            if reg == 0x184:
                return
        # Timers (0x0020a000)
        elif (phys & 0xfffff000) == 0x0020a000:
            reg = phys & 0xfff
            c_base = (phys & 0xffffffc0) >> 2
            c_reg = reg & 0x3f
            if c_reg == 0x00:
                if (val & 3) != 0:
                    mmio_low_u32[idx] = np.uint32(0)
                    c_idx = (reg >> 6) & 3
                    if c_idx < 3:
                        b_addr = 0x00500060 + 10 + c_idx
                        ch = (mmio_low_u32[b_addr >> 2] >> ((b_addr & 3) << 3)) & np.uint32(0x3f)
                        if ch < 32:
                            state[ST_IRQ_RAW_LO] &= ~(np.uint64(1) << np.uint64(ch))
                        else:
                            state[ST_IRQ_RAW_HI] &= ~(np.uint64(1) << np.uint64(ch - 32))
                return
            elif c_reg == 0x10:
                mmio_low_u32[idx] = val
                if (val & 3) != 0:
                    ini = mmio_low_u32[c_base + (0x28 >> 2)]
                    mmio_low_u32[c_base + (0x04 >> 2)] = ini
                return
            elif c_reg == 0x28:
                mmio_low_u32[idx] = val
                mmio_low_u32[c_base + (0x04 >> 2)] = val
                return
        # USB OHCI0 (0x00900000) & OHCI1 (0x00901000)
        elif (phys & 0xffffe000) == 0x00900000:
            reg = phys & 0xfff
            base_idx = (phys & 0xfffff000) >> 2
            if reg == 0x08:
                if (val & 1) != 0:
                    mmio_low_u32[base_idx + (0x34 >> 2)] = np.uint32(0x2edf)
                mmio_low_u32[idx] = np.uint32(val & np.uint32(0xfffffffe))
                return
        # USB EHCI (0x00904000)
        elif (phys & 0xfffff000) == 0x00904000:
            reg = phys & 0xfff
            if reg == 0x10:
                mmio_low_u32[idx] = np.uint32(val & np.uint32(0xfffffffd))
                return
        # DesignWare SPI (0x00e00000)
        elif (phys & 0xfffff000) == 0x00e00000:
            reg = phys & 0xfff
            if reg == 0x08:
                mmio_low_u32[idx] = val
                if (val & 1) == 0:
                    state[ST_SPI_RX_AVAIL] = 0
                return
            elif reg == 0x18 or reg == 0x1c:
                mmio_low_u32[idx] = np.uint32(val & 0x1f)
                return
            elif reg == 0x60:
                ctrlr0 = mmio_low_u32[(0x00e00000) >> 2]
                dfs = ctrlr0 & 0x1f
                tmod = (ctrlr0 >> 10) & 3
                if tmod == 0:
                    tmod = (ctrlr0 >> 8) & 3
                if tmod == 2:
                    ndf = np.uint64(mmio_low_u32[(0x00e00004) >> 2] & 0xffff)
                    state[ST_SPI_RX_AVAIL] = ndf + np.uint64(1)
                    state[ST_SPI_JUST_DATA] = 0
                else:
                    if dfs == 31:
                        _spi_tx_byte((val >> 24) & 0xff, state)
                        _spi_tx_byte((val >> 16) & 0xff, state)
                        _spi_tx_byte((val >> 8) & 0xff, state)
                        _spi_tx_byte(val & 0xff, state)
                    elif dfs == 15:
                        _spi_tx_byte((val >> 8) & 0xff, state)
                        _spi_tx_byte(val & 0xff, state)
                    else:
                        _spi_tx_byte(val & 0xff, state)
                    if tmod == 0:
                        state[ST_SPI_RX_AVAIL] += np.uint64(1)
                return
        # OTP (0x00f80080)
        elif (phys & 0xfffffff0) == 0x00f80080:
            if (phys & 0xf) == 0x8:
                return
        mmio_low_u32[idx] = val
    elif 0x04000000 <= phys < 0x05000000:
        vpu_idx = (phys - 0x04000000) >> 2
        if phys == 0x04700004:
            mmio_vpu_u32[vpu_idx] = val
            mmio_vpu_u32[(0x0070000c) >> 2] = val
            return
        elif phys == 0x04700018:
            prev = mmio_vpu_u32[vpu_idx]
            mmio_vpu_u32[vpu_idx] = val
            if (val & np.uint32(0x10000)) != 0 and (prev & np.uint32(0x10000)) == 0:
                cmd_cnt = val & np.uint32(0xffff)
                base_phys = mmio_vpu_u32[(0x00700004) >> 2]
                buf_sz = mmio_vpu_u32[(0x00700008) >> 2]
                cur_phys = mmio_vpu_u32[(0x0070000c) >> 2]
                if cur_phys == np.uint32(0):
                    cur_phys = base_phys
                off = (cur_phys - base_phys) + cmd_cnt * np.uint32(80)
                if buf_sz > np.uint32(0) and off >= buf_sz:
                    off = off % buf_sz
                mmio_vpu_u32[(0x0070000c) >> 2] = base_phys + off
                mmio_vpu_u32[(0x0070001c) >> 2] |= np.uint32(0x10)
                if (mmio_vpu_u32[(0x0070001c) >> 2] & np.uint32(1)) != 0:
                    b_addr = 0x00500060 + 62
                    ch = (mmio_low_u32[b_addr >> 2] >> ((b_addr & 3) << 3)) & np.uint32(0x3f)
                    if ch < 32:
                        state[ST_IRQ_RAW_LO] |= (np.uint64(1) << np.uint64(ch))
                    else:
                        state[ST_IRQ_RAW_HI] |= (np.uint64(1) << np.uint64(ch - 32))
            return
        elif phys == 0x0470001c:
            if (val & np.uint32(2)) != 0:
                b_addr = 0x00500060 + 62
                ch = (mmio_low_u32[b_addr >> 2] >> ((b_addr & 3) << 3)) & np.uint32(0x3f)
                if ch < 32:
                    state[ST_IRQ_RAW_LO] &= ~(np.uint64(1) << np.uint64(ch))
                else:
                    state[ST_IRQ_RAW_HI] &= ~(np.uint64(1) << np.uint64(ch - 32))
            mmio_vpu_u32[vpu_idx] = (val & ~np.uint32(2)) | np.uint32(0x10)
            return
        mmio_vpu_u32[vpu_idx] = val


@njit( cache=True )
def _read32(addr, state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash):
    phys = np.uint32(addr) & np.uint32(0x1fffffff)
    if 0x10000000 <= phys < 0x14000000:
        off = phys - np.uint32(0x10000000)
        if (off & 3) == 0:
            return dram_u32[off >> 2]
        return np.uint32(
            np.uint32(dram[off])
            | (np.uint32(dram[off + 1]) << 8)
            | (np.uint32(dram[off + 2]) << 16)
            | (np.uint32(dram[off + 3]) << 24)
        )
    elif 0x00100000 <= phys < 0x00110000:
        if (phys & 3) == 0:
            return mmio_low_u32[phys >> 2]
        return np.uint32(
            np.uint32(mmio_low[phys])
            | (np.uint32(mmio_low[phys + 1]) << 8)
            | (np.uint32(mmio_low[phys + 2]) << 16)
            | (np.uint32(mmio_low[phys + 3]) << 24)
        )
    return _mmio_read32(phys, state, mmio_low_u32, mmio_vpu_u32, flash)


@njit( cache=True )
def _read16(addr, state, dram, dram_u16, mmio_low, mmio_low_u16, mmio_low_u32, mmio_vpu_u32, flash):
    phys = np.uint32(addr) & np.uint32(0x1fffffff)
    if 0x10000000 <= phys < 0x14000000:
        off = phys - np.uint32(0x10000000)
        if (off & 1) == 0:
            return np.uint32(dram_u16[off >> 1])
        return np.uint32(np.uint32(dram[off]) | (np.uint32(dram[off + 1]) << 8))
    elif 0x00100000 <= phys < 0x00110000:
        if (phys & 1) == 0:
            return np.uint32(mmio_low_u16[phys >> 1])
        return np.uint32(np.uint32(mmio_low[phys]) | (np.uint32(mmio_low[phys + 1]) << 8))
    val32 = _mmio_read32(phys & np.uint32(0xfffffffc), state, mmio_low_u32, mmio_vpu_u32, flash)
    shift = (phys & 2) << 3
    return np.uint32((val32 >> shift) & 0xffff)


@njit( cache=True )
def _read8(addr, state, dram, mmio_low, mmio_low_u32, mmio_vpu_u32, flash):
    phys = np.uint32(addr) & np.uint32(0x1fffffff)
    if 0x10000000 <= phys < 0x14000000:
        return np.uint32(dram[phys - np.uint32(0x10000000)])
    elif 0x00100000 <= phys < 0x00110000:
        return np.uint32(mmio_low[phys])
    elif 0x00000000 <= phys < 0x01000000 and (phys & 0xfffff000) == 0x0030a000:
        return np.uint32(mmio_low[phys])
    val32 = _mmio_read32(phys & np.uint32(0xfffffffc), state, mmio_low_u32, mmio_vpu_u32, flash)
    shift = (phys & 3) << 3
    return np.uint32((val32 >> shift) & 0xff)


@njit( cache=True )
def _write32(addr, val_in, state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log):
    phys = np.uint32(addr) & np.uint32(0x1fffffff)
    val = np.uint32(val_in)
    if 0x10000000 <= phys < 0x14000000:
        off = phys - np.uint32(0x10000000)
        if (off & 3) == 0:
            dram_u32[off >> 2] = val
        else:
            dram[off] = np.uint8(val & 0xff)
            dram[off + 1] = np.uint8((val >> 8) & 0xff)
            dram[off + 2] = np.uint8((val >> 16) & 0xff)
            dram[off + 3] = np.uint8((val >> 24) & 0xff)
        return
    elif 0x00100000 <= phys < 0x00110000:
        if (phys & 3) == 0:
            mmio_low_u32[phys >> 2] = val
        else:
            mmio_low[phys] = np.uint8(val & 0xff)
            mmio_low[phys + 1] = np.uint8((val >> 8) & 0xff)
            mmio_low[phys + 2] = np.uint8((val >> 16) & 0xff)
            mmio_low[phys + 3] = np.uint8((val >> 24) & 0xff)
        return
    if phys == 0x04700018:
        _ga_exec_cmds(val, state, dram, dram_u32, mmio_vpu_u32, mmio_log)
    _mmio_write32(phys, val, state, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)


@njit( cache=True )
def _write16(addr, val, state, dram, dram_u16, mmio_low, mmio_low_u16, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log):
    phys = np.uint32(addr) & np.uint32(0x1fffffff)
    val16 = np.uint16(val & 0xffff)
    if 0x10000000 <= phys < 0x14000000:
        off = phys - np.uint32(0x10000000)
        if (off & 1) == 0:
            dram_u16[off >> 1] = val16
        else:
            dram[off] = np.uint8(val16 & 0xff)
            dram[off + 1] = np.uint8((val16 >> 8) & 0xff)
        return
    elif 0x00100000 <= phys < 0x00110000:
        if (phys & 1) == 0:
            mmio_low_u16[phys >> 1] = val16
        else:
            mmio_low[phys] = np.uint8(val16 & 0xff)
            mmio_low[phys + 1] = np.uint8((val16 >> 8) & 0xff)
        return
    elif 0x00000000 <= phys < 0x01000000:
        mmio_low_u16[phys >> 1] = val16
        return
    _mmio_write32(phys & np.uint32(0xfffffffc), np.uint32(val16), state, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)


@njit( cache=True )
def _write8(addr, val, state, dram, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log):
    phys = np.uint32(addr) & np.uint32(0x1fffffff)
    val8 = np.uint8(val & 0xff)
    if 0x10000000 <= phys < 0x14000000:
        dram[phys - np.uint32(0x10000000)] = val8
        return
    elif 0x00100000 <= phys < 0x00110000:
        mmio_low[phys] = val8
        return
    elif (phys & 0xffffe000) == 0x00402000 and (phys & 0xfff) == 0x00:
        _mmio_write32(phys, np.uint32(val8), state, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
        return
    elif 0x00000000 <= phys < 0x01000000:
        mmio_low[phys] = val8
        return
    elif 0x04000000 <= phys < 0x05000000:
        off = phys - np.uint32(0x04000000)
        w = mmio_vpu_u32[off >> 2]
        sh = (off & 3) << 3
        w = np.uint32((w & ~(np.uint32(0xff) << sh)) | (np.uint32(val8) << sh))
        mmio_vpu_u32[off >> 2] = w


@njit( cache=True )
def run_cpu(
    max_steps,
    regs,
    cr,
    state,
    dram,
    dram_u16,
    dram_u32,
    mmio_low,
    mmio_low_u16,
    mmio_low_u32,
    mmio_vpu_u32,
    flash,
    uart_buf,
    mmio_log,
    stop_on_stage_transition,
):
    pc = np.uint32(state[ST_PC])
    hi = np.uint32(state[ST_HI])
    lo = np.uint32(state[ST_LO])
    c_flag = np.uint32(cr[0] & 1)
    insn_count = np.uint64(state[ST_INSN_COUNT])
    state[ST_STOP_REASON] = STOP_RUNNING

    for step in range(max_steps):
        if (step & 0xff) == 0:
            delta = np.uint32(16) if (state[ST_IG_SYNCED] == np.uint64(2) or state[ST_KEY_CODE] != np.uint64(0)) else np.uint32(256)
            if (0x90107ea8 <= pc <= 0x90107ebc or pc == 0x90103294) and (cr[0] & 0x40) != 0:
                c0_val_cur = mmio_low_u32[(0x0020a004) >> 2]
                rem_to_ovf = np.uint32(np.uint32(0xffffffff) - c0_val_cur + np.uint32(1))
                if rem_to_ovf > delta and rem_to_ovf <= np.uint32(20000):
                    delta = rem_to_ovf

            c0_cfg = mmio_low_u32[(0x0020a020) >> 2]
            if (c0_cfg & 1) != 0:
                c0_val = mmio_low_u32[(0x0020a004) >> 2]
                c0_new = np.uint32(c0_val + delta)
                if c0_new < c0_val:
                    c0_ini = mmio_low_u32[(0x0020a028) >> 2]
                    c0_new = c0_ini
                    if (c0_cfg & 2) != 0:
                        mmio_low_u32[(0x0020a000) >> 2] = np.uint32(1)
                        ch = np.uint32(mmio_low[0x00500060 + 10]) & np.uint32(0x3f)
                        if ch < 32:
                            state[ST_IRQ_RAW_LO] |= np.uint64(1) << np.uint64(ch)
                        else:
                            state[ST_IRQ_RAW_HI] |= np.uint64(1) << np.uint64(ch - 32)
                mmio_low_u32[(0x0020a004) >> 2] = c0_new

            c1_cfg = mmio_low_u32[(0x0020a060) >> 2]
            if (c1_cfg & 1) != 0:
                c1_val = mmio_low_u32[(0x0020a044) >> 2]
                c1_new = np.uint32(c1_val + delta)
                if c1_new < c1_val:
                    c1_new = mmio_low_u32[(0x0020a068) >> 2]
                    if (c1_cfg & 2) != 0:
                        mmio_low_u32[(0x0020a040) >> 2] = np.uint32(1)
                        ch = np.uint32(mmio_low[0x00500060 + 11]) & np.uint32(0x3f)
                        if ch < 32:
                            state[ST_IRQ_RAW_LO] |= np.uint64(1) << np.uint64(ch)
                        else:
                            state[ST_IRQ_RAW_HI] |= np.uint64(1) << np.uint64(ch - 32)
                mmio_low_u32[(0x0020a044) >> 2] = c1_new

            c2_cfg = mmio_low_u32[(0x0020a0a0) >> 2]
            if (c2_cfg & 1) != 0:
                c2_val = mmio_low_u32[(0x0020a084) >> 2]
                c2_new = np.uint32(c2_val + delta)
                if c2_new < c2_val:
                    c2_new = mmio_low_u32[(0x0020a0a8) >> 2]
                    if (c2_cfg & 2) != 0:
                        mmio_low_u32[(0x0020a080) >> 2] = np.uint32(1)
                        ch = np.uint32(mmio_low[0x00500060 + 12]) & np.uint32(0x3f)
                        if ch < 32:
                            state[ST_IRQ_RAW_LO] |= np.uint64(1) << np.uint64(ch)
                        else:
                            state[ST_IRQ_RAW_HI] |= np.uint64(1) << np.uint64(ch - 32)
                mmio_low_u32[(0x0020a084) >> 2] = c2_new

            if (cr[0] & 0x40) != 0:
                pen_lo = np.uint32(state[ST_IRQ_RAW_LO]) & (~mmio_low_u32[(0x00500050) >> 2])
                pen_hi = np.uint32(state[ST_IRQ_RAW_HI]) & (~mmio_low_u32[(0x00500054) >> 2])
                if pen_lo != 0 or pen_hi != 0:
                    cr[0] = np.uint32((cr[0] & np.uint32(0xfffffffe)) | (c_flag & np.uint32(1)))
                    cr[2] = cr[0]
                    cr[4] = pc
                    cr[0] = np.uint32((cr[0] & np.uint32(0xff80feaf)) | np.uint32(0x800a0000))
                    c_flag = np.uint32(cr[0] & np.uint32(1))
                    vbr = cr[1]
                    pc = np.uint32(_read32(np.uint32(vbr + np.uint32(40)), state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash) & np.uint32(0xfffffffe))

        if (pc & np.uint32(0xdc000000)) == np.uint32(0x90000000):
            inst = np.uint32(dram_u16[(pc & np.uint32(0x03ffffff)) >> np.uint32(1)])
        else:
            phys_pc = np.uint32(pc & np.uint32(0x1fffffff))
            if 0x00100000 <= phys_pc < 0x00110000:
                inst = np.uint32(mmio_low_u16[phys_pc >> 1])
            else:
                state[ST_PC] = pc
                state[ST_STOP_REASON] = STOP_PC_OOB
                break

        state[ST_LAST_PC] = pc
        next_pc = np.uint32(pc + np.uint32(2))
        insn_count += np.uint64(1)

        op = np.uint32((inst >> 12) & 0xf)
        rx = np.uint32(inst & 0xf)
        ry = np.uint32((inst >> 4) & 0xf)
        rz = np.uint32((inst >> 8) & 0xf)

        if op == 0x0:
            if rz == 0x0:
                if ry == 0x0:
                    if inst == 0x0000:  # bkpt
                        state[ST_PC] = pc
                        state[ST_STOP_REASON] = STOP_BKPT
                        break
                    elif inst == 0x0001:  # sync
                        pass
                    elif inst == 0x0002:  # rte
                        cr[0] = cr[2]
                        c_flag = np.uint32(cr[0] & np.uint32(1))
                        next_pc = np.uint32(cr[4] & np.uint32(0xfffffffe))
                    elif inst == 0x0003:  # rfi
                        cr[0] = cr[3]
                        c_flag = np.uint32(cr[0] & np.uint32(1))
                        next_pc = np.uint32(cr[5] & np.uint32(0xfffffffe))
                    elif inst == 0x0004 or inst == 0x0005 or inst == 0x0006:
                        pass
                    elif inst == 0x0007:
                        pass
                    elif 0x0008 <= inst <= 0x000b:  # trap 0..3
                        vec = np.uint32(16 + (inst & 3))
                        cr[0] = np.uint32((cr[0] & np.uint32(0xfffffffe)) | (c_flag & np.uint32(1)))
                        cr[2] = cr[0]
                        cr[4] = next_pc
                        cr[0] = np.uint32((cr[0] & np.uint32(0xff80feaf)) | np.uint32(0x80000000) | (vec << np.uint32(16)))
                        c_flag = np.uint32(cr[0] & np.uint32(1))
                        next_pc = np.uint32(_read32(np.uint32(cr[1] + (vec << np.uint32(2))), state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash) & np.uint32(0xfffffffe))
                elif ry == 0x1:  # cpseti
                    pass
                elif ry == 0x2:  # mvc rx
                    regs[rx] = np.uint32(c_flag & np.uint32(1))
                elif ry == 0x3:  # mvcv rx
                    regs[rx] = np.uint32((c_flag ^ np.uint32(1)) & np.uint32(1))
                elif ry == 0x4:  # ldq r4-r7, (rx)
                    base_addr = regs[rx]
                    r4 = _read32(base_addr, state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
                    r5 = _read32(np.uint32(base_addr + np.uint32(4)), state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
                    r6 = _read32(np.uint32(base_addr + np.uint32(8)), state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
                    r7 = _read32(np.uint32(base_addr + np.uint32(12)), state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
                    regs[4] = r4
                    regs[5] = r5
                    regs[6] = r6
                    regs[7] = r7
                elif ry == 0x5:  # stq r4-r7, (rx)
                    base_addr = regs[rx]
                    _write32(base_addr, regs[4], state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
                    _write32(np.uint32(base_addr + np.uint32(4)), regs[5], state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
                    _write32(np.uint32(base_addr + np.uint32(8)), regs[6], state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
                    _write32(np.uint32(base_addr + np.uint32(12)), regs[7], state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
                elif ry == 0x6:  # ldm rf-r15, (r0)
                    base_addr = regs[0]
                    offset = np.uint32(0)
                    for r_idx in range(rx, 16):
                        regs[r_idx] = _read32(np.uint32(base_addr + offset), state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
                        offset = np.uint32(offset + np.uint32(4))
                elif ry == 0x7:  # stm rf-r15, (r0)
                    base_addr = regs[0]
                    offset = np.uint32(0)
                    for r_idx in range(rx, 16):
                        _write32(np.uint32(base_addr + offset), regs[r_idx], state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
                        offset = np.uint32(offset + np.uint32(4))
                elif ry == 0x8:  # dect rx
                    if c_flag != 0:
                        regs[rx] = np.uint32(regs[rx] - np.uint32(1))
                elif ry == 0x9:  # decf rx
                    if c_flag == 0:
                        regs[rx] = np.uint32(regs[rx] - np.uint32(1))
                elif ry == 0xa:  # inct rx
                    if c_flag != 0:
                        regs[rx] = np.uint32(regs[rx] + np.uint32(1))
                elif ry == 0xb:  # incf rx
                    if c_flag == 0:
                        regs[rx] = np.uint32(regs[rx] + np.uint32(1))
                elif ry == 0xc:  # jmp rx
                    next_pc = np.uint32(regs[rx] & np.uint32(0xfffffffe))
                elif ry == 0xd:  # jsr rx
                    target = np.uint32(regs[rx] & np.uint32(0xfffffffe))
                    regs[15] = next_pc
                    next_pc = target
                elif ry == 0xe:  # ff1 rx
                    regs[rx] = _ff1_32(regs[rx])
                elif ry == 0xf:  # brev rx
                    regs[rx] = _brev32(regs[rx])
            elif rz == 0x1:
                v = regs[rx]
                if ry == 0x0:  # xtrb3 r1, rx
                    regs[1] = np.uint32(v & np.uint32(0xff))
                elif ry == 0x1:  # xtrb2 r1, rx
                    regs[1] = np.uint32((v >> np.uint32(8)) & np.uint32(0xff))
                elif ry == 0x2:  # xtrb1 r1, rx
                    regs[1] = np.uint32((v >> np.uint32(16)) & np.uint32(0xff))
                elif ry == 0x3:  # xtrb0 r1, rx
                    regs[1] = np.uint32((v >> np.uint32(24)) & np.uint32(0xff))
                elif ry == 0x4:  # zextb rx
                    regs[rx] = np.uint32(v & np.uint32(0xff))
                elif ry == 0x5:  # sextb rx
                    b = np.uint32(v & np.uint32(0xff))
                    regs[rx] = np.uint32(b | np.uint32(0xffffff00)) if (b & np.uint32(0x80)) != 0 else b
                elif ry == 0x6:  # zexth rx
                    regs[rx] = np.uint32(v & np.uint32(0xffff))
                elif ry == 0x7:  # sexth rx
                    h = np.uint32(v & np.uint32(0xffff))
                    regs[rx] = np.uint32(h | np.uint32(0xffff0000)) if (h & np.uint32(0x8000)) != 0 else h
                elif ry == 0x8:  # declt rx
                    v2 = np.uint32(v - np.uint32(1))
                    regs[rx] = v2
                    c_flag = np.uint32(1) if (v2 & np.uint32(0x80000000)) != 0 else np.uint32(0)
                elif ry == 0x9:  # tstnbz rx
                    b0 = v & np.uint32(0xff)
                    b1 = (v >> np.uint32(8)) & np.uint32(0xff)
                    b2 = (v >> np.uint32(16)) & np.uint32(0xff)
                    b3 = (v >> np.uint32(24)) & np.uint32(0xff)
                    c_flag = np.uint32(1) if (b0 != 0 and b1 != 0 and b2 != 0 and b3 != 0) else np.uint32(0)
                elif ry == 0xa:  # decgt rx
                    v2 = np.uint32(v - np.uint32(1))
                    regs[rx] = v2
                    c_flag = np.uint32(1) if np.int32(v2) > 0 else np.uint32(0)
                elif ry == 0xb:  # decne rx
                    v2 = np.uint32(v - np.uint32(1))
                    regs[rx] = v2
                    c_flag = np.uint32(1) if v2 != 0 else np.uint32(0)
                elif ry == 0xc:  # clrt rx
                    if c_flag != 0:
                        regs[rx] = np.uint32(0)
                elif ry == 0xd:  # clrf rx
                    if c_flag == 0:
                        regs[rx] = np.uint32(0)
                elif ry == 0xe:  # abs rx
                    if (v & np.uint32(0x80000000)) != 0:
                        regs[rx] = np.uint32(np.uint32(0) - v)
                elif ry == 0xf:  # not rx
                    regs[rx] = np.uint32(~v)
            elif rz == 0x2:  # movt rx, ry
                if c_flag != 0:
                    regs[rx] = regs[ry]
            elif rz == 0x3:  # mult rx, ry
                prod = np.uint64(regs[rx]) * np.uint64(regs[ry])
                regs[rx] = np.uint32(prod & np.uint64(0xffffffff))
                lo = np.uint32(prod & np.uint64(0xffffffff))
                hi = np.uint32(prod >> np.uint64(32))
            elif rz == 0x4:  # mac rx, ry
                regs[rx] = np.uint32(regs[rx] + regs[ry] * regs[1])
            elif rz == 0x5:  # subu rx, ry
                regs[rx] = np.uint32(regs[rx] - regs[ry])
            elif rz == 0x6:  # addc rx, ry
                s = np.uint64(regs[rx]) + np.uint64(regs[ry]) + np.uint64(c_flag)
                regs[rx] = np.uint32(s & np.uint64(0xffffffff))
                c_flag = np.uint32((s >> np.uint64(32)) & np.uint64(1))
            elif rz == 0x7:  # subc rx, ry
                borrow = np.int64(1) - np.int64(c_flag)
                diff = np.int64(regs[rx]) - np.int64(regs[ry]) - borrow
                regs[rx] = np.uint32(diff & np.int64(0xffffffff))
                c_flag = np.uint32(1) if diff >= 0 else np.uint32(0)
            elif rz == 0xa:  # movf rx, ry
                if c_flag == 0:
                    regs[rx] = regs[ry]
            elif rz == 0xb:  # lsr rx, ry
                sh = np.uint32(regs[ry] & np.uint32(0x3f))
                regs[rx] = np.uint32(regs[rx] >> sh) if sh < 32 else np.uint32(0)
            elif rz == 0xc:  # cmphs rx, ry
                c_flag = np.uint32(1) if regs[rx] >= regs[ry] else np.uint32(0)
            elif rz == 0xd:  # cmplt rx, ry
                c_flag = np.uint32(1) if np.int32(regs[rx]) < np.int32(regs[ry]) else np.uint32(0)
            elif rz == 0xe:  # tst rx, ry
                c_flag = np.uint32(1) if (regs[rx] & regs[ry]) != 0 else np.uint32(0)
            elif rz == 0xf:  # cmpne rx, ry
                c_flag = np.uint32(1) if regs[rx] != regs[ry] else np.uint32(0)

        elif op == 0x1:
            if rz == 0x0 or rz == 0x1:
                if (inst & 0xfff8) == 0x11f0:  # psrclr imm3
                    imm3 = np.uint32(inst & 7)
                    if imm3 == 0:
                        cr[0] = np.uint32(cr[0] & ~np.uint32(0x02))
                    else:
                        mask = np.uint32(((imm3 & 1) << 6) | (((imm3 >> 1) & 1) << 4) | (((imm3 >> 2) & 1) << 8))
                        cr[0] = np.uint32(cr[0] & ~mask)
                elif (inst & 0xfff8) == 0x11f8:  # psrset imm3
                    imm3 = np.uint32(inst & 7)
                    if imm3 == 0:
                        cr[0] = np.uint32(cr[0] | np.uint32(0x02))
                    else:
                        mask = np.uint32(((imm3 & 1) << 6) | (((imm3 >> 1) & 1) << 4) | (((imm3 >> 2) & 1) << 8))
                        cr[0] = np.uint32(cr[0] | mask)
                else:  # mfcr rx, cr[imm5]
                    imm5 = np.uint32((inst >> 4) & 0x1f)
                    if imm5 == 0:
                        cr[0] = np.uint32((cr[0] & np.uint32(0xfffffffe)) | (c_flag & np.uint32(1)))
                    regs[rx] = cr[imm5]
            elif rz == 0x8 or rz == 0x9:  # mtcr rx, cr[imm5]
                imm5 = np.uint32((inst >> 4) & 0x1f)
                cr[imm5] = regs[rx]
                if imm5 == 0:
                    c_flag = np.uint32(cr[0] & np.uint32(1))
            elif rz == 0x2:  # mov rx, ry
                regs[rx] = regs[ry]
            elif rz == 0x3:  # bgenr rx, ry
                regs[rx] = np.uint32(np.uint32(1) << (regs[ry] & np.uint32(0x1f)))
            elif rz == 0x4:  # rsub rx, ry
                regs[rx] = np.uint32(regs[ry] - regs[rx])
            elif rz == 0x5:  # ixw rx, ry
                regs[rx] = np.uint32(regs[rx] + (regs[ry] << np.uint32(2)))
            elif rz == 0x6:  # and rx, ry
                regs[rx] = np.uint32(regs[rx] & regs[ry])
            elif rz == 0x7:  # xor rx, ry
                regs[rx] = np.uint32(regs[rx] ^ regs[ry])
            elif rz == 0xa:  # asr rx, ry
                sh = np.uint32(regs[ry] & np.uint32(0x3f))
                if sh >= 32:
                    regs[rx] = np.uint32(0xffffffff) if (regs[rx] & np.uint32(0x80000000)) != 0 else np.uint32(0)
                else:
                    regs[rx] = np.uint32(np.int32(regs[rx]) >> np.int32(sh))
            elif rz == 0xb:  # lsl rx, ry
                sh = np.uint32(regs[ry] & np.uint32(0x3f))
                regs[rx] = np.uint32(regs[rx] << sh) if sh < 32 else np.uint32(0)
            elif rz == 0xc:  # addu rx, ry
                regs[rx] = np.uint32(regs[rx] + regs[ry])
            elif rz == 0xd:  # ixh rx, ry
                regs[rx] = np.uint32(regs[rx] + (regs[ry] << np.uint32(1)))
            elif rz == 0xe:  # or rx, ry
                regs[rx] = np.uint32(regs[rx] | regs[ry])
            elif rz == 0xf:  # andn rx, ry
                regs[rx] = np.uint32(regs[rx] & (~regs[ry]))

        elif op == 0x2:
            sub = np.uint32((inst >> 9) & 0x7)
            imm5 = np.uint32((inst >> 4) & 0x1f)
            if sub == 0:  # addi rx, oimm5
                regs[rx] = np.uint32(regs[rx] + imm5 + np.uint32(1))
            elif sub == 1:  # cmplti rx, oimm5
                c_flag = np.uint32(1) if np.int32(regs[rx]) < np.int32(imm5 + np.uint32(1)) else np.uint32(0)
            elif sub == 2:  # subi rx, oimm5
                regs[rx] = np.uint32(regs[rx] - (imm5 + np.uint32(1)))
            elif sub == 3:  # cpwgr
                pass
            elif sub == 4:  # rsubi rx, imm5
                regs[rx] = np.uint32(imm5 - regs[rx])
            elif sub == 5:  # cmpnei rx, imm5
                c_flag = np.uint32(1) if regs[rx] != imm5 else np.uint32(0)
            elif sub == 6:
                if imm5 == 0:  # bmaski rx, 32
                    regs[rx] = np.uint32(0xffffffff)
                elif imm5 == 1:  # divu rx, r1
                    if regs[1] != 0:
                        regs[rx] = np.uint32(regs[rx] // regs[1])
                elif imm5 == 2 or imm5 == 6:  # mflos / mflo
                    regs[rx] = lo
                elif imm5 == 3 or imm5 == 7:  # mfhis / mfhi
                    regs[rx] = hi
                elif imm5 == 4:  # mtlo
                    lo = regs[rx]
                elif imm5 == 5:  # mthi
                    hi = regs[rx]
                else:  # bmaski rx, imm5
                    regs[rx] = np.uint32((np.uint32(1) << imm5) - np.uint32(1))
            elif sub == 7:  # andi rx, imm5
                regs[rx] = np.uint32(regs[rx] & imm5)

        elif op == 0x3:
            sub = np.uint32((inst >> 9) & 0x7)
            imm5 = np.uint32((inst >> 4) & 0x1f)
            if sub == 0:  # bclri rx, imm5
                regs[rx] = np.uint32(regs[rx] & ~(np.uint32(1) << imm5))
            elif sub == 1:
                if imm5 == 1:  # divs rx, r1
                    d = np.int64(np.int32(regs[1]))
                    if d != 0:
                        n = np.int64(np.int32(regs[rx]))
                        q = np.int64(n // d)
                        if (n ^ d) < 0 and (q * d != n):
                            q += np.int64(1)
                        regs[rx] = np.uint32(np.int32(q))
                elif imm5 >= 4:  # bgeni rx, imm5
                    regs[rx] = np.uint32(np.uint32(1) << imm5)
            elif sub == 2:  # bseti rx, imm5
                regs[rx] = np.uint32(regs[rx] | (np.uint32(1) << imm5))
            elif sub == 3:  # btsti rx, imm5
                c_flag = np.uint32((regs[rx] >> imm5) & np.uint32(1))
            elif sub == 4:
                if imm5 == 0:  # xsr rx
                    v = regs[rx]
                    old_c = np.uint32(c_flag & np.uint32(1))
                    c_flag = np.uint32(v & np.uint32(1))
                    regs[rx] = np.uint32((v >> np.uint32(1)) | (old_c << np.uint32(31)))
                else:  # rotli rx, imm5
                    v = regs[rx]
                    regs[rx] = np.uint32((v << imm5) | (v >> (np.uint32(32) - imm5)))
            elif sub == 5:
                if imm5 == 0:  # asrc rx
                    v = regs[rx]
                    c_flag = np.uint32(v & np.uint32(1))
                    regs[rx] = np.uint32((v >> np.uint32(1)) | (v & np.uint32(0x80000000)))
                else:  # asri rx, imm5
                    regs[rx] = np.uint32(np.int32(regs[rx]) >> np.int32(imm5))
            elif sub == 6:
                if imm5 == 0:  # lslc rx
                    v = regs[rx]
                    c_flag = np.uint32((v >> np.uint32(31)) & np.uint32(1))
                    regs[rx] = np.uint32(v << np.uint32(1))
                else:  # lsli rx, imm5
                    regs[rx] = np.uint32(regs[rx] << imm5)
            elif sub == 7:
                if imm5 == 0:  # lsrc rx
                    v = regs[rx]
                    c_flag = np.uint32(v & np.uint32(1))
                    regs[rx] = np.uint32(v >> np.uint32(1))
                else:  # lsri rx, imm5
                    regs[rx] = np.uint32(regs[rx] >> imm5)

        elif op == 0x5:
            if rz == 0x0:  # muls rx, ry
                prod_s = np.int64(np.int32(regs[rx])) * np.int64(np.int32(regs[ry]))
                lo = np.uint32(prod_s & np.int64(0xffffffff))
                hi = np.uint32((prod_s >> np.int64(32)) & np.int64(0xffffffff))
            elif rz == 0x1:  # mulsa rx, ry
                acc = (np.int64(np.int32(hi)) << np.int64(32)) | np.int64(np.uint64(lo))
                acc += np.int64(np.int32(regs[rx])) * np.int64(np.int32(regs[ry]))
                lo = np.uint32(acc & np.int64(0xffffffff))
                hi = np.uint32((acc >> np.int64(32)) & np.int64(0xffffffff))
            elif rz == 0x2:  # mulss rx, ry
                acc = (np.int64(np.int32(hi)) << np.int64(32)) | np.int64(np.uint64(lo))
                acc -= np.int64(np.int32(regs[rx])) * np.int64(np.int32(regs[ry]))
                lo = np.uint32(acc & np.int64(0xffffffff))
                hi = np.uint32((acc >> np.int64(32)) & np.int64(0xffffffff))
            elif rz == 0x4:  # mulu rx, ry
                prod_u = np.uint64(regs[rx]) * np.uint64(regs[ry])
                lo = np.uint32(prod_u & np.uint64(0xffffffff))
                hi = np.uint32(prod_u >> np.uint64(32))
            elif rz == 0x5:  # mulua rx, ry
                acc_u = (np.uint64(hi) << np.uint64(32)) | np.uint64(lo)
                acc_u += np.uint64(regs[rx]) * np.uint64(regs[ry])
                lo = np.uint32(acc_u & np.uint64(0xffffffff))
                hi = np.uint32(acc_u >> np.uint64(32))
            elif rz == 0x6:  # mulus rx, ry
                acc_u = (np.uint64(hi) << np.uint64(32)) | np.uint64(lo)
                acc_u -= np.uint64(regs[rx]) * np.uint64(regs[ry])
                lo = np.uint32(acc_u & np.uint64(0xffffffff))
                hi = np.uint32(acc_u >> np.uint64(32))

        elif op == 0x6:
            if rz < 8:  # movi rx, imm7
                imm7 = np.uint32((inst >> 4) & 0x7f)
                regs[rx] = imm7
            elif rz == 8:  # mulsh rx, ry
                p = np.int32(np.int16(regs[rx] & np.uint32(0xffff))) * np.int32(np.int16(regs[ry] & np.uint32(0xffff)))
                regs[rx] = np.uint32(p)
                lo = np.uint32(p)

        elif op == 0x7:
            imm8 = np.uint32(inst & 0xff)
            pool_addr = np.uint32((pc + np.uint32(2) + (imm8 << np.uint32(2))) & np.uint32(0xfffffffc))
            if (pool_addr & np.uint32(0xdc000000)) == np.uint32(0x90000000):
                val = dram_u32[(pool_addr & np.uint32(0x03ffffff)) >> np.uint32(2)]
            else:
                val = _read32(pool_addr, state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
            if rz == 0:  # jmpi
                next_pc = np.uint32(val & np.uint32(0xfffffffe))
            elif rz == 15:  # jsri
                regs[15] = next_pc
                next_pc = np.uint32(val & np.uint32(0xfffffffe))
            else:  # lrw rz, [pool_addr]
                regs[rz] = val

        elif op == 0x8:  # ld.w rz, (rx, imm4*4)
            addr = np.uint32(regs[rx] + (ry << np.uint32(2)))
            if (addr & np.uint32(0xdc000003)) == np.uint32(0x90000000):
                regs[rz] = dram_u32[(addr & np.uint32(0x03ffffff)) >> np.uint32(2)]
            else:
                regs[rz] = _read32(addr, state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
        elif op == 0x9:  # st.w rz, (rx, imm4*4)
            addr = np.uint32(regs[rx] + (ry << np.uint32(2)))
            if (addr & np.uint32(0xdc000003)) == np.uint32(0x90000000):
                dram_u32[(addr & np.uint32(0x03ffffff)) >> np.uint32(2)] = regs[rz]
            else:
                _write32(addr, regs[rz], state, dram, dram_u32, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
        elif op == 0xa:  # ld.b rz, (rx, imm4)
            addr = np.uint32(regs[rx] + ry)
            if (addr & np.uint32(0xdc000000)) == np.uint32(0x90000000):
                regs[rz] = np.uint32(dram[addr & np.uint32(0x03ffffff)])
            else:
                regs[rz] = _read8(addr, state, dram, mmio_low, mmio_low_u32, mmio_vpu_u32, flash)
        elif op == 0xb:  # st.b rz, (rx, imm4)
            addr = np.uint32(regs[rx] + ry)
            if (addr & np.uint32(0xdc000000)) == np.uint32(0x90000000):
                dram[addr & np.uint32(0x03ffffff)] = np.uint8(regs[rz] & np.uint32(0xff))
            else:
                _write8(addr, regs[rz], state, dram, mmio_low, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)
        elif op == 0xc:  # ld.h rz, (rx, imm4*2)
            addr = np.uint32(regs[rx] + (ry << np.uint32(1)))
            if (addr & np.uint32(0xdc000001)) == np.uint32(0x90000000):
                regs[rz] = np.uint32(dram_u16[(addr & np.uint32(0x03ffffff)) >> np.uint32(1)])
            else:
                regs[rz] = _read16(addr, state, dram, dram_u16, mmio_low, mmio_low_u16, mmio_low_u32, mmio_vpu_u32, flash)
        elif op == 0xd:  # st.h rz, (rx, imm4*2)
            addr = np.uint32(regs[rx] + (ry << np.uint32(1)))
            if (addr & np.uint32(0xdc000001)) == np.uint32(0x90000000):
                dram_u16[(addr & np.uint32(0x03ffffff)) >> np.uint32(1)] = np.uint16(regs[rz] & np.uint32(0xffff))
            else:
                _write16(addr, regs[rz], state, dram, dram_u16, mmio_low, mmio_low_u16, mmio_low_u32, mmio_vpu_u32, uart_buf, mmio_log)

        elif op == 0xe:  # bt / bf
            off11 = np.int32(inst & 0x7ff)
            if (off11 & 0x400) != 0:
                off11 -= np.int32(0x800)
            cond = (c_flag == 0) if (inst & 0x0800) != 0 else (c_flag != 0)
            if cond:
                next_pc = np.uint32(np.int32(pc) + np.int32(2) + (off11 << np.int32(1)))

        elif op == 0xf:  # br / bsr
            off11 = np.int32(inst & 0x7ff)
            if (off11 & 0x400) != 0:
                off11 -= np.int32(0x800)
            target = np.uint32(np.int32(pc) + np.int32(2) + (off11 << np.int32(1)))
            if (inst & 0x0800) != 0:  # bsr
                regs[15] = next_pc
            elif off11 == -1 and (cr[0] & 0x40) == 0:
                state[ST_PC] = pc
                state[ST_STOP_REASON] = STOP_INFINITE_LOOP
                break
            next_pc = target

        hi_pc = next_pc >> np.uint32(16)
        if hi_pc == np.uint32(0x9010):
            if next_pc == np.uint32(0x90107ec8):
                next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
            elif next_pc == np.uint32(0x90106b50):  # memset(r2, r3, r4)
                dst_p = regs[2] & np.uint32(0x1fffffff)
                n_b = regs[4]
                if np.uint32(0x10000000) <= dst_p and dst_p + n_b <= np.uint32(0x14000000):
                    off_d = dst_p - np.uint32(0x10000000)
                    dram[off_d : off_d + n_b] = np.uint8(regs[3] & np.uint32(0xff))
                    next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
            elif next_pc == np.uint32(0x90106a74):  # memcpy(r2, r3, r4)
                dst_p = regs[2] & np.uint32(0x1fffffff)
                src_p = regs[3] & np.uint32(0x1fffffff)
                n_b = regs[4]
                if (
                    np.uint32(0x10000000) <= dst_p
                    and dst_p + n_b <= np.uint32(0x14000000)
                    and np.uint32(0x10000000) <= src_p
                    and src_p + n_b <= np.uint32(0x14000000)
                ):
                    off_d = dst_p - np.uint32(0x10000000)
                    off_s = src_p - np.uint32(0x10000000)
                    dram[off_d : off_d + n_b] = dram[off_s : off_s + n_b]
                    next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
            elif insn_count >= np.uint64(8000000) or state[ST_IG_SYNCED] != np.uint64(0):
                if next_pc == np.uint32(0x90103274):  # hal_lsbit_index(r2)
                    v_lsb = regs[2]
                    if v_lsb == 0:
                        regs[2] = np.uint32(0xffffffff)
                    else:
                        regs[2] = np.uint32(31) - _ff1_32(v_lsb & np.uint32(np.uint32(0) - v_lsb))
                    next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
                elif next_pc == np.uint32(0x9010cb78):  # strcasecmp(r2, r3)
                    s1_p = (regs[2] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    s2_p = (regs[3] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    if s1_p < np.uint32(0x03fff000) and s2_p < np.uint32(0x03fff000):
                        while True:
                            c1 = np.uint32(dram[s1_p])
                            c2 = np.uint32(dram[s2_p])
                            if 65 <= c1 <= 90:
                                c1 += np.uint32(32)
                            if 65 <= c2 <= 90:
                                c2 += np.uint32(32)
                            if c1 == 0 or c1 != c2:
                                regs[2] = np.uint32(c1 - c2)
                                break
                            s1_p += np.uint32(1)
                            s2_p += np.uint32(1)
                        next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
                elif next_pc == np.uint32(0x9010cbc4):  # strncasecmp(r2, r3, r4)
                    n_cmp = regs[4]
                    s1_p = (regs[2] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    s2_p = (regs[3] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    if s1_p < np.uint32(0x03fff000) and s2_p < np.uint32(0x03fff000):
                        res_cmp = np.uint32(0)
                        for _ in range(n_cmp):
                            c1 = np.uint32(dram[s1_p])
                            c2 = np.uint32(dram[s2_p])
                            if 65 <= c1 <= 90:
                                c1 += np.uint32(32)
                            if 65 <= c2 <= 90:
                                c2 += np.uint32(32)
                            res_cmp = np.uint32(c1 - c2)
                            if c1 == 0 or c2 == 0 or res_cmp != 0:
                                break
                            s1_p += np.uint32(1)
                            s2_p += np.uint32(1)
                        regs[2] = res_cmp
                        next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
                elif next_pc == np.uint32(0x9010c570) or next_pc == np.uint32(0x9010c5e4):  # strcmp(r2, r3)
                    s1_p = (regs[2] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    s2_p = (regs[3] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    if s1_p < np.uint32(0x03fff000) and s2_p < np.uint32(0x03fff000):
                        while True:
                            c1 = np.uint32(dram[s1_p])
                            c2 = np.uint32(dram[s2_p])
                            if c1 == 0 or c1 != c2:
                                regs[2] = np.uint32(c1 - c2)
                                break
                            s1_p += np.uint32(1)
                            s2_p += np.uint32(1)
                        next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
                elif next_pc == np.uint32(0x9010c690):  # strlen(r2)
                    s1_p = (regs[2] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    if s1_p < np.uint32(0x03fff000):
                        cur_p = s1_p
                        while dram[cur_p] != 0:
                            cur_p += np.uint32(1)
                        regs[2] = np.uint32(cur_p - s1_p)
                        next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
        elif hi_pc == np.uint32(0x901c):
            if state[ST_KEY_CODE] != 0:
                if next_pc == np.uint32(0x901c2e62):  # Main GUI loop event poll entry
                    if (state[ST_KEY_CODE] >> np.uint64(63)) != 0 and dram[0x5a6920] == 0:
                        state[ST_KEY_CODE] = np.uint64(0)
                        pc = next_pc
                        break
                elif next_pc == np.uint32(0x901c2e64):  # Main GUI loop event injection (0x905a6920)
                    if (state[ST_KEY_CODE] >> np.uint64(63)) == 0 and dram[0x5a6920] == 0:
                        sc = np.uint32(state[ST_KEY_CODE] & np.uint64(0xffffffff))
                        sym = np.uint16((state[ST_KEY_CODE] >> np.uint64(32)) & np.uint64(0xffff))
                        dram[0x5a6920] = np.uint8(2)
                        dram_u32[0x5a6924 >> 2] = sc
                        dram_u16[0x5a6928 >> 1] = sym
                        state[ST_KEY_CODE] = np.uint64(1) << np.uint64(63)
                elif next_pc == np.uint32(0x901c9d0a):  # Modal popup dialog loop poll entry
                    if (state[ST_KEY_CODE] >> np.uint64(63)) != 0:
                        state[ST_KEY_CODE] = np.uint64(0)
                        pc = next_pc
                        break
                elif next_pc == np.uint32(0x901c9d0c):  # Modal popup dialog loop key injection
                    if (state[ST_KEY_CODE] >> np.uint64(63)) == 0:
                        sc = np.uint32(state[ST_KEY_CODE] & np.uint64(0xffffffff))
                        sym = np.uint16((state[ST_KEY_CODE] >> np.uint64(32)) & np.uint64(0xffff))
                        ev_off = (regs[2] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                        dram[ev_off] = np.uint8(2)
                        dram_u32[(ev_off + np.uint32(4)) >> 2] = sc
                        dram_u16[(ev_off + np.uint32(8)) >> 1] = sym
                        state[ST_KEY_CODE] = np.uint64(1) << np.uint64(63)
                        regs[2] = np.uint32(2)
                        next_pc = np.uint32(0x901c9d0e)
        elif hi_pc == np.uint32(0x903b):
            if insn_count >= np.uint64(8000000) or state[ST_IG_SYNCED] != np.uint64(0):
                if next_pc == np.uint32(0x903b941c):  # __muldi3
                    a64 = (np.uint64(regs[3]) << np.uint64(32)) | np.uint64(regs[2])
                    b64 = (np.uint64(regs[5]) << np.uint64(32)) | np.uint64(regs[4])
                    p64 = np.uint64(a64 * b64)
                    regs[2] = np.uint32(p64 & np.uint64(0xffffffff))
                    regs[3] = np.uint32(p64 >> np.uint64(32))
                    next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
                elif next_pc == np.uint32(0x903bafb4):  # __udivdi3
                    n64 = (np.uint64(regs[3]) << np.uint64(32)) | np.uint64(regs[2])
                    d64 = (np.uint64(regs[5]) << np.uint64(32)) | np.uint64(regs[4])
                    if d64 != 0:
                        q64 = np.uint64(n64 // d64)
                        regs[2] = np.uint32(q64 & np.uint64(0xffffffff))
                        regs[3] = np.uint32(q64 >> np.uint64(32))
                        next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
                elif next_pc == np.uint32(0x903bb388):  # __umoddi3
                    n64 = (np.uint64(regs[3]) << np.uint64(32)) | np.uint64(regs[2])
                    d64 = (np.uint64(regs[5]) << np.uint64(32)) | np.uint64(regs[4])
                    if d64 != 0:
                        r64 = np.uint64(n64 % d64)
                        regs[2] = np.uint32(r64 & np.uint64(0xffffffff))
                        regs[3] = np.uint32(r64 >> np.uint64(32))
                        next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
        elif hi_pc == np.uint32(0x9039):
            if next_pc == np.uint32(0x9039ef00):
                next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
            elif next_pc == np.uint32(0x90397788) and (insn_count >= np.uint64(8000000) or state[ST_IG_SYNCED] != np.uint64(0)):
                if state[ST_SPI_STATE] == 5 and regs[12] >= np.uint32(4):
                    n_b4 = regs[12] & np.uint32(0xfffffffc)
                    dst_off = (regs[7] & np.uint32(0x1fffffff)) - np.uint32(0x10000000)
                    f_addr = np.uint32(state[ST_SPI_ADDR] & np.uint64(0x3fffff))
                    if dst_off < np.uint32(0x04000000) and dst_off + n_b4 <= np.uint32(0x04000000) and f_addr + n_b4 <= np.uint32(0x400000):
                        dram[dst_off : dst_off + n_b4] = flash[f_addr : f_addr + n_b4]
                        state[ST_SPI_ADDR] = np.uint64((f_addr + n_b4) & np.uint32(0x3fffff))
                        regs[6] = regs[12] & np.uint32(3)
                        regs[9] = np.uint32(regs[7] + n_b4)
                        regs[5] = np.uint32(3)
                        next_pc = np.uint32(0x903977ba)
        elif hi_pc == np.uint32(0x901d):
            if next_pc == np.uint32(0x901de454) and (insn_count >= np.uint64(8000000) or state[ST_IG_SYNCED] != np.uint64(0)):
                w_row = dram_u32[((regs[12] & np.uint32(0x03ffffff)) + np.uint32(32)) >> np.uint32(2)]
                src_off = (regs[3] & np.uint32(0x03ffffff)) + regs[6] * w_row + regs[5]
                dst_off = (regs[7] & np.uint32(0x03ffffff)) >> np.uint32(2)
                rem_x = np.int32(w_row) - np.int32(regs[5])
                if rem_x > 0 and src_off + np.uint32(rem_x) <= np.uint32(0x04000000) and (dst_off + np.uint32(rem_x)) <= np.uint32(0x01000000):
                    for x_i in range(rem_x):
                        dram_u32[dst_off + np.uint32(x_i)] |= np.uint32(dram[src_off + np.uint32(x_i)]) << np.uint32(24)
                    regs[4] = w_row
                    regs[5] = w_row
                    regs[7] = np.uint32(regs[7] + (np.uint32(rem_x) << np.uint32(2)))
                    c_flag = np.uint32(0)
                    next_pc = np.uint32(0x901de472)
        elif hi_pc == np.uint32(0x900c):
            if next_pc == np.uint32(0x900c68c6):  # VPU_FLIP_SURFACE -> trigger VPU VSync ISR (0x900c6c38)
                regs[15] = np.uint32(0x900c68ce)
                next_pc = np.uint32(0x900c6c38)
        elif hi_pc == np.uint32(0x9000):
            if next_pc == np.uint32(0x90009f4c):  # Installation Guide timeout timer (stop after first sync draw)
                if state[ST_IG_SYNCED] == np.uint64(0):
                    state[ST_IG_SYNCED] = np.uint64(1)
                elif state[ST_IG_SYNCED] == np.uint64(10):
                    state[ST_IG_SYNCED] = np.uint64(11)
                else:
                    if state[ST_IG_SYNCED] == np.uint64(11):
                        state[ST_KEY_CODE] = np.uint64(1) << np.uint64(63)
                    state[ST_IG_SYNCED] = np.uint64(2)
                    regs[2] = np.uint32(0)
                    next_pc = np.uint32(regs[15] & np.uint32(0xfffffffe))
            elif stop_on_stage_transition and next_pc == np.uint32(0x90000000) and pc >= np.uint32(0x93000000):
                pc = next_pc
                state[ST_STOP_REASON] = STOP_STAGE3_ENTRY
                break
        elif hi_pc == np.uint32(0x9036):
            if next_pc == np.uint32(0x9036f6ac):
                pc = next_pc
                state[ST_STOP_REASON] = STOP_LZMA_DECODE
                break
        elif hi_pc == np.uint32(0x9316):
            if next_pc == np.uint32(0x9316e744):
                pc = next_pc
                state[ST_STOP_REASON] = STOP_LZMA_DECODE
                break
            elif stop_on_stage_transition and next_pc == np.uint32(0x9316b400) and pc < np.uint32(0x90000000):
                pc = next_pc
                state[ST_STOP_REASON] = STOP_STAGE2_ENTRY
                break

        pc = next_pc
    else:
        state[ST_STOP_REASON] = STOP_MAX_STEPS

    cr[0] = np.uint32((cr[0] & np.uint32(0xfffffffe)) | (c_flag & np.uint32(1)))
    state[ST_PC] = pc
    state[ST_HI] = hi
    state[ST_LO] = lo
    state[ST_INSN_COUNT] = insn_count
    return state[ST_STOP_REASON]


class GX6706Emulator:
    # Remote control & front-panel button keymap from /dvb/theme/keymap.xml & /dvb/theme/panel.xml
    # Maps button name -> (raw_scancode, guik_symbol)
    REMOTE_KEYS = {
        "UP": (0x7f65, 0x111),
        "DOWN": (0x7f55, 0x112),
        "LEFT": (0x7f4d, 0x114),
        "RIGHT": (0x7f85, 0x113),
        "OK": (0x7f45, 0x00d),
        "ENTER": (0x7f45, 0x00d),
        "RETURN": (0x7f45, 0x00d),
        "EXIT": (0x7ffd, 0x01b),
        "ESC": (0x7ffd, 0x01b),
        "ESCAPE": (0x7ffd, 0x01b),
        "MENU": (0x7fcd, 0x06d),
        "RED": (0x7f95, 0x190),
        "GREEN": (0x7fa5, 0x194),
        "YELLOW": (0x7fbd, 0x191),
        "BLUE": (0x7f9d, 0x193),
        "0": (0x7fa7, 0x030),
        "1": (0x7f37, 0x031),
        "2": (0x7f27, 0x032),
        "3": (0x7f1f, 0x033),
        "4": (0x7f17, 0x034),
        "5": (0x7f07, 0x035),
        "6": (0x7f3f, 0x036),
        "7": (0x7f97, 0x037),
        "8": (0x7f87, 0x038),
        "9": (0x7fbf, 0x039),
        "PAGE_UP": (0x7ff5, 0x118),
        "CH+": (0x7ff5, 0x118),
        "PAGE_DOWN": (0x7fed, 0x119),
        "CH-": (0x7fed, 0x119),
        "VOL+": (0xbb1d, 0x02b),
        "VOL-": (0xbb47, 0x02d),
        "MUTE": (0x7f35, 0x068),
        "INFO": (0x7fdd, 0x069),
        "EPG": (0x7fd5, 0x065),
        "FAV": (0x7f9f, 0x066),
        "AUDIO": (0x7f8f, 0x061),
        "SUBTITLE": (0x7f79, 0x073),
        "TELETEXT": (0xbbc7, 0x074),
        "TV_RADIO": (0x7fc5, 0x072),
        "POWER": (0x7f75, 0x076),
        # Front-panel physical button aliases (/dev/gxpanel0 & /dvb/theme/panel.xml)
        "PANEL_MENU": (0xbbcf, 0x06d),
        "PANEL_EXIT": (0xbb4f, 0x01b),
        "PANEL_ESC": (0xbb4f, 0x01b),
        "PANEL_OK": (0xbb57, 0x00d),
        "PANEL_ENTER": (0xbb57, 0x00d),
        "PANEL_UP": (0xbb77, 0x111),
        "PANEL_CH+": (0xbb77, 0x111),
        "PANEL_DOWN": (0xbbd7, 0x112),
        "PANEL_CH-": (0xbbd7, 0x112),
        "PANEL_LEFT": (0xbbb7, 0x114),
        "PANEL_VOL-": (0xbbb7, 0x114),
        "PANEL_RIGHT": (0xbb37, 0x113),
        "PANEL_VOL+": (0xbb37, 0x113),
        "PANEL_POWER": (0xbbaf, 0x068),
        "PANEL_RED": (0xbb67, 0x190),
        "PANEL_GREEN": (0xbba7, 0x194),
        "PANEL_YELLOW": (0xbb27, 0x191),
        "PANEL_BLUE": (0xbbc7, 0x193),
    }

    # Front-panel buttons configured by app_get_panel_keymap (0x9050bed8..0x9050bef4) & /dvb/theme/panel.xml
    PANEL_KEYS = {
        "MENU": (0xbbcf, 0x06d),
        "EXIT": (0xbb4f, 0x01b),
        "ESC": (0xbb4f, 0x01b),
        "OK": (0xbb57, 0x00d),
        "ENTER": (0xbb57, 0x00d),
        "UP": (0xbb77, 0x111),
        "CH+": (0xbb77, 0x111),
        "DOWN": (0xbbd7, 0x112),
        "CH-": (0xbbd7, 0x112),
        "LEFT": (0xbbb7, 0x114),
        "VOL-": (0xbbb7, 0x114),
        "RIGHT": (0xbb37, 0x113),
        "VOL+": (0xbb37, 0x113),
        "POWER": (0xbbaf, 0x068),
        "RED": (0xbb67, 0x190),
        "GREEN": (0xbba7, 0x194),
        "YELLOW": (0xbb27, 0x191),
        "BLUE": (0xbbc7, 0x193),
    }

    MAIN_MENU_TABS = ("Program", "Picture", "Search", "Time", "Option", "System", "USB")

    def __init__(self, flash_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ZenWire_DVB-T2-001-GX6706_4mb.bin")):
        self.flash_path = flash_path
        if os.path.exists(flash_path):
            with open(flash_path, "rb") as f:
                flash_bytes = f.read()
            self.flash = np.frombuffer(flash_bytes, dtype=np.uint8).copy()
        else:
            self.flash = np.zeros(4 * 1024 * 1024, dtype=np.uint8)

        self.dram = np.zeros(64 * 1024 * 1024, dtype=np.uint8)
        self.dram_u16 = self.dram.view(np.uint16)
        self.dram_u32 = self.dram.view(np.uint32)

        self.mmio_low = np.zeros(16 * 1024 * 1024, dtype=np.uint8)
        self.mmio_low_u16 = self.mmio_low.view(np.uint16)
        self.mmio_low_u32 = self.mmio_low.view(np.uint32)

        self.mmio_vpu = np.zeros(16 * 1024 * 1024, dtype=np.uint8)
        self.mmio_vpu_u32 = self.mmio_vpu.view(np.uint32)

        self.regs = np.zeros(16, dtype=np.uint32)
        self.cr = np.zeros(32, dtype=np.uint32)
        self.state = np.zeros(20, dtype=np.uint64)

        self.uart_buf = np.zeros(256 * 1024, dtype=np.uint8)
        self.mmio_log = np.zeros((16384, 3), dtype=np.uint32)

        self.reset()

    def reset(self):
        self.regs.fill(0)
        self.cr.fill(0)
        self.state.fill(0)
        self.mmio_low.fill(0)
        self.mmio_vpu.fill(0)

        # Supervisor mode (S=1) in PSR (cr0)
        self.cr[0] = 0x80000000

        # Hardware reset defaults for GX6705/GX6706 (Cygnus)
        # 1. Chip ID at 0x0030a184: (val >> 14) & 0xffff == 0x6705
        self.mmio_low_u32[0x0030a184 >> 2] = 0x6705 << 14
        # 2. Reversed chip name at 0x0030a190..0x0030a19b: bytes 8..11 = "5076" -> reverses to "6705"
        self.mmio_low[0x0030a198:0x0030a19c] = np.frombuffer(b"5076", dtype=np.uint8)
        # 3. DDR controller calibration done bit 5 at 0x00c000b8
        self.mmio_low_u32[0x00c000b8 >> 2] = 0x00000020
        # 4. OTP controller ready bits 9 & 10 at 0x00f80088
        self.mmio_low_u32[0x00f80088 >> 2] = 0x00000600
        # 5. DesignWare SPI version "102*" at 0x00e0005c
        self.mmio_low_u32[0x00e0005c >> 2] = 0x3130322a
        # 6. Video PHY PLL locked at 0x00702018 / 0x00702418
        self.mmio_low_u32[0x00702018 >> 2] = 0xffffffff
        self.mmio_low_u32[0x00702418 >> 2] = 0xffffffff
        # 7. Default INTC masks
        self.mmio_low_u32[0x00500050 >> 2] = 0xffffffff
        self.mmio_low_u32[0x00500054 >> 2] = 0xffffffff
        # Default 1-to-1 INTC channel mapping
        for i in range(64):
            self.mmio_low[0x00500060 + i] = i
        # 8. USB OHCI0 (0x00900000), OHCI1 (0x00901000), EHCI (0x00904000) defaults
        for ohci_base in (0x00900000, 0x00901000):
            self.mmio_low_u32[(ohci_base + 0x00) >> 2] = 0x00000110  # HcRevision
            self.mmio_low_u32[(ohci_base + 0x34) >> 2] = 0x27782edf  # HcFmInterval
            self.mmio_low_u32[(ohci_base + 0x48) >> 2] = 0x02000001  # HcRhDescriptorA (1 port)
        self.mmio_low_u32[0x00904000 >> 2] = 0x01000010  # CAPLENGTH=0x10, HCIVERSION=0x0100
        self.mmio_low_u32[0x00904004 >> 2] = 0x00010012  # HCSPARAMS: 2 ports
        self.mmio_low_u32[0x00904008 >> 2] = 0x00000006  # HCCPARAMS
        self.mmio_low_u32[0x00904010 >> 2] = 0x00080000  # USBCMD
        self.mmio_low_u32[0x00904014 >> 2] = 0x00001000  # USBSTS: STS_HALT

    def load_stage1(self):
        """BootROM loads flash[0x0004:0x2004] to SRAM at 0x00100000 and stores Stage 2 flash offset/size."""
        self.reset()
        # Copy 8KB of Stage 1 after the 4-byte 0xaa55aa55 header into SRAM at 0x00100000
        self.mmio_low[0x00100000:0x00102000] = self.flash[0x0004:0x2004]
        self.state[ST_PC] = 0x00100000

    def load_stage3_fast(self, ecos_bin: bytes = None):
        """Fast-boot helper: load decompressed ecos.bin directly at 0x90000000 with GxLoader ATAG bootargs."""
        self.reset()
        if ecos_bin is None:
            ecos_bin = lzma.decompress(bytes(self.flash[0x30080:0x30080 + 2304104]))
        self.dram[:len(ecos_bin)] = np.frombuffer(ecos_bin, dtype=np.uint8)
        # Write 8-byte ATAG header + bootargs string into DRAM at 0x93186318 (exact layout GxLoader builds)
        atag_bootargs = (
            b"\x00\x02\x00\x00\x03\x00AT"
            b"mem=22169K@0x10000000 videomem=40M@0x115a6400 femem=2407K@0x13da6400 "
            b"mem_end console=ttyS0,115200 init=/init root=/dev/mtdblock4 "
            b"mtdparts=m25p80:128k@0m(BOOT),512@128k(TABLE),65024@131584(LOGO),"
            b"2304k@192k(KERNEL),448k@2496k(ROOT),1152k@2944k(DATA) mtdparts_end \x00"
        )
        arg_off = 0x03186318
        self.dram[arg_off:arg_off + len(atag_bootargs)] = np.frombuffer(atag_bootargs, dtype=np.uint8)
        self.regs[2] = 0xa2a25441
        self.regs[3] = 0x93186318
        self.cr[8] = 0xa2a25441
        self.cr[9] = 0x93186318
        # Set PLL registers as Stage 1 configures them so clock calculations in eCos match
        self.mmio_low_u32[0x0030a0c0 >> 2] = 0x46cf7100
        self.mmio_low_u32[0x0030a0c4 >> 2] = 0x1fc4ef40
        self.mmio_low_u32[0x0030a0c8 >> 2] = 0x27c19cc0
        self.mmio_low_u32[0x0030a178 >> 2] = 0x05285239
        self.state[ST_PC] = 0x90000000

    def step(self, max_steps=100000, stop_on_stage_transition=False):
        start_insns = int(self.state[ST_INSN_COUNT])
        while True:
            elapsed = int(self.state[ST_INSN_COUNT]) - start_insns
            rem = max_steps - elapsed
            if rem <= 0:
                return STOP_MAX_STEPS
            res = int(
                run_cpu(
                    rem,
                    self.regs,
                    self.cr,
                    self.state,
                    self.dram,
                    self.dram_u16,
                    self.dram_u32,
                    self.mmio_low,
                    self.mmio_low_u16,
                    self.mmio_low_u32,
                    self.mmio_vpu_u32,
                    self.flash,
                    self.uart_buf,
                    self.mmio_log,
                    stop_on_stage_transition,
                )
            )
            if res == STOP_LZMA_DECODE:
                if int(self.state[ST_PC]) == 0x9036f6ac:
                    dst_off = (int(self.regs[3]) & 0x1fffffff) - 0x10000000
                    src_off = (int(self.regs[4]) & 0x1fffffff) - 0x10000000
                    src_len = int(self.regs[6])
                    raw_xz = bytes(self.dram[src_off:src_off + src_len])
                    decompressed = lzma.decompress(raw_xz)
                    self.dram[dst_off:dst_off + len(decompressed)] = np.frombuffer(decompressed, dtype=np.uint8)
                    self.regs[2] = np.uint32(len(decompressed))
                    self.state[ST_PC] = np.uint64(self.regs[15])
                    continue
                src_off = (int(self.regs[2]) & 0x1fffffff) - 0x10000000
                src_len = int(self.regs[3])
                dst_off = (int(self.regs[4]) & 0x1fffffff) - 0x10000000
                raw_lzma = bytes(self.dram[src_off:src_off + src_len])
                decompressed = lzma.decompress(raw_lzma)
                self.dram[dst_off:dst_off + len(decompressed)] = np.frombuffer(decompressed, dtype=np.uint8)
                self.regs[2] = np.uint32(0)
                self.state[ST_PC] = np.uint64(self.regs[15])
                continue
            return res

    def get_uart_output(self) -> str:
        n = int(self.state[ST_UART_TX_LEN])
        return bytes(self.uart_buf[:n]).decode("latin1")

    def get_osd_region_info(self) -> dict:
        """Read the active Cygnus VPU OSD region descriptor pointed to by 0x04804004."""
        head_phys = int(self.mmio_vpu_u32[0x00804004 >> 2])
        if head_phys == 0:
            return {"enabled": False, "head_phys": 0, "width": 1024, "height": 576, "fb_phys": 0x11f950b8}
        off_w = (head_phys & 0x03ffffff) >> 2
        w0 = int(self.dram_u32[off_w + 0])
        w2 = int(self.dram_u32[off_w + 2])
        w3 = int(self.dram_u32[off_w + 3])
        w4 = int(self.dram_u32[off_w + 4])
        w6 = int(self.dram_u32[off_w + 6])
        width = ((w2 >> 16) & 0xfff) - (w2 & 0xfff) + 1
        height = ((w3 >> 16) & 0xfff) - (w3 & 0xfff) + 1
        stride = w6 & 0x1fff
        return {
            "enabled": bool(self.mmio_vpu_u32[0x00804000 >> 2] & 1),
            "head_phys": head_phys,
            "word0": w0,
            "width": width,
            "height": height,
            "stride": stride,
            "fb_phys": w4,
            "colorkey_reg": int(self.mmio_vpu_u32[0x00804010 >> 2]),
        }

    def get_vpu_flip_count(self) -> int:
        """Return the number of VPU OSD surface page flips completed since boot."""
        vpu_state = int(self.dram_u32[0x5180a8 >> 2])
        if 0x90000000 <= vpu_state < 0x94000000:
            return int(self.dram_u32[(vpu_state + 0x3c8 - 0x90000000) >> 2])
        return 0

    def _read_cstr(self, ptr: int, max_len: int = 64) -> str:
        if not (0x90000000 <= ptr < 0x94000000):
            return ""
        off = ptr - 0x90000000
        raw = bytes(self.dram[off : off + max_len]).split(b"\x00")[0]
        return raw.decode("latin1", errors="replace")

    def get_focus_path(self) -> list:
        """Return the active GUI widget focus chain from top-level window down to focused child widget."""
        curr = int(self.dram_u32[0x5a6850 >> 2])
        chain = []
        seen = set()
        while 0x90000000 <= curr < 0x94000000 and curr not in seen:
            seen.add(curr)
            off_w = (curr - 0x90000000) >> 2
            name = self._read_cstr(int(self.dram_u32[off_w + 0]))
            if name:
                chain.append(name)
            curr = int(self.dram_u32[off_w + 23])  # +92: parent pointer
        chain.reverse()

        curr = int(self.dram_u32[0x5a6850 >> 2])
        if 0x90000000 <= curr < 0x94000000:
            child = int(self.dram_u32[((curr - 0x90000000) >> 2) + 24])  # +96: first_child
            while 0x90000000 <= child < 0x94000000 and child not in seen:
                seen.add(child)
                c_off = (child - 0x90000000) >> 2
                focused = int(self.dram_u32[c_off + 33])  # +132: focused flag
                if focused != 0:
                    cname = self._read_cstr(int(self.dram_u32[c_off + 0]))
                    if cname:
                        chain.append(cname)
                    child = int(self.dram_u32[c_off + 24])
                else:
                    child = int(self.dram_u32[c_off + 27])  # +108: next_sibling
        return chain

    def get_active_window(self) -> str:
        """Return the name of the currently active/focused GUI widget or window."""
        curr = int(self.dram_u32[0x5a6850 >> 2])
        if 0x90000000 <= curr < 0x94000000:
            return self._read_cstr(int(self.dram_u32[(curr - 0x90000000) >> 2]))
        return ""

    def get_main_menu_tab(self) -> str:
        """Return the currently selected Main Menu tab name ('Program', 'Picture', 'Search', 'Time', 'Option', 'System', 'USB')."""
        idx = int(self.dram_u32[(0x9050e2cc - 0x90000000) >> 2])
        if 0 <= idx < len(self.MAIN_MENU_TABS):
            return self.MAIN_MENU_TABS[idx]
        return str(idx)

    def render_osd(self, output_path: str = None, transparent_to_black: bool = False) -> np.ndarray:
        """Decode the active 16bpp RGB565 OSD framebuffer to an (H, W, 3) uint8 RGB image."""
        info = self.get_osd_region_info()
        w = info["width"]
        h = info["height"]
        fb_off = (info["fb_phys"] & 0x03ffffff) >> 1
        fb16 = self.dram_u16[fb_off : fb_off + w * h].reshape((h, w))
        r = (((fb16 >> 11) & 0x1f).astype(np.uint32) * 255 // 31).astype(np.uint8)
        g = (((fb16 >> 5) & 0x3f).astype(np.uint32) * 255 // 63).astype(np.uint8)
        b = ((fb16 & 0x1f).astype(np.uint32) * 255 // 31).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        if transparent_to_black:
            mask = fb16 == 0x8df7
            rgb[mask] = 0
        if output_path is not None:
            from PIL import Image
            Image.fromarray(rgb).save(output_path)
        return rgb

    def press_key(self, key, max_steps: int = 60_000_000) -> int:
        """
        Emulate pressing a remote control or front-panel button.
        `key` can be a button name in REMOTE_KEYS (e.g. 'UP', 'DOWN', 'LEFT', 'RIGHT',
        'OK', 'EXIT', 'MENU', 'RED', 'GREEN', 'YELLOW', 'BLUE', '0'..'9', 'PANEL_MENU', etc.)
        or a (scancode, guik_sym) tuple.
        """
        if isinstance(key, str):
            k_norm = key.strip().upper()
            if k_norm not in self.REMOTE_KEYS:
                raise ValueError(f"Unknown key {key!r}. Valid keys: {sorted(self.REMOTE_KEYS.keys())}")
            scancode, guik_sym = self.REMOTE_KEYS[k_norm]
        elif isinstance(key, tuple):
            scancode, guik_sym = int(key[0]) & 0xffffffff, int(key[1]) & 0xffff
        else:
            scancode = int(key) & 0xffffffff
            guik_sym = 0
            for sc, sym in self.REMOTE_KEYS.values():
                if sc == scancode:
                    guik_sym = sym
                    break

        flips_before = self.get_vpu_flip_count()
        self.state[ST_KEY_CODE] = (np.uint64(guik_sym) << np.uint64(32)) | np.uint64(scancode)
        if max_steps <= 0:
            return STOP_RUNNING

        start_insns = int(self.state[ST_INSN_COUNT])
        while True:
            rem = max_steps - (int(self.state[ST_INSN_COUNT]) - start_insns)
            if rem <= 0:
                self.state[ST_KEY_CODE] = np.uint64(0)
                return STOP_MAX_STEPS
            res = self.step(max_steps=rem)
            if self.state[ST_KEY_CODE] == 0:
                info = self.get_osd_region_info()
                fb_off = (info["fb_phys"] & 0x03ffffff) >> 1
                fb_cur = self.dram_u16[fb_off : fb_off + info["width"] * info["height"]]
                non_bg_px = int(np.count_nonzero((fb_cur != 0) & (fb_cur != 0x8df7)))
                if self.get_vpu_flip_count() > flips_before and non_bg_px >= 1000:
                    return res
                # Advance one more GUI event loop iteration if a non-blank frame hasn't flipped yet
                self.state[ST_KEY_CODE] = np.uint64(1) << np.uint64(63)
            elif res != STOP_RUNNING:
                return res

    def press_panel_button(self, button: str, max_steps: int = 60_000_000) -> int:
        """
        Emulate pressing a physical front-panel button (/dev/gxpanel0 & /dvb/theme/panel.xml).
        `button` can be 'MENU', 'EXIT', 'OK', 'UP'/'CH+', 'DOWN'/'CH-', 'LEFT'/'VOL-', 'RIGHT'/'VOL+', 'POWER',
        or 'RED', 'GREEN', 'YELLOW', 'BLUE'.
        """
        b_norm = button.strip().upper()
        if b_norm.startswith("PANEL_"):
            b_norm = b_norm[6:]
        if b_norm not in self.PANEL_KEYS:
            raise ValueError(f"Unknown panel button {button!r}. Valid buttons: {sorted(self.PANEL_KEYS.keys())}")
        return self.press_key(self.PANEL_KEYS[b_norm], max_steps=max_steps)

    def send_keys(self, keys, max_steps_per_key: int = 60_000_000):
        """Send a sequence of remote control / front-panel button presses."""
        for k in keys:
            self.press_key(k, max_steps=max_steps_per_key)

    def boot_to_gui(self, max_steps: int = 110_000_000) -> int:
        """Fast-boot Stage 3 and run until the initial Installation Guide OSD is rendered and synced."""
        if int(self.state[ST_PC]) == 0:
            self.load_stage3_fast()
        self.state[ST_IG_SYNCED] = np.uint64(10)
        return self.step(max_steps=max_steps, stop_on_stage_transition=False)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="GX6706 (NationalChip Cygnus C-SKY v1) Full-System Emulator")
    parser.add_argument("--fast", action="store_true", help="Fast-boot directly into Stage 3 GUI (skip Stage 1/2 SPI copy)")
    parser.add_argument("--keys", type=str, default="", help="Comma-separated sequence of remote/panel keys to press after boot (e.g. EXIT,RIGHT,OK)")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "osd_output.png"), help="Output PNG path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    emu = GX6706Emulator()
    if args.fast:
        emu.boot_to_gui()
    else:
        emu.load_stage1()
        # Stage 1 -> Stage 2 -> Stage 3 -> OSD UI
        emu.step(max_steps=5_000_000, stop_on_stage_transition=True)
        emu.step(max_steps=50_000_000, stop_on_stage_transition=True)
        emu.step(max_steps=135_000_000, stop_on_stage_transition=False)

    if args.keys:
        key_list = [k.strip() for k in args.keys.split(",") if k.strip()]
        emu.send_keys(key_list)

    elapsed = time.perf_counter() - t0
    total_insns = int(emu.state[ST_INSN_COUNT])
    out_png = args.output
    emu.render_osd(output_path=out_png)
    osd_info = emu.get_osd_region_info()

    print(f"=== GX6706 3-Stage Boot & OSD Render Complete ===")
    print(f"Total Instructions : {total_insns:,} ({total_insns / elapsed / 1e6:.2f} MIPS in {elapsed:.2f}s)")
    print(f"Active GUI Focus   : {emu.get_focus_path()} (Tab: {emu.get_main_menu_tab()})")
    print(f"OSD Region         : {osd_info['width']}x{osd_info['height']} @ phys 0x{osd_info['fb_phys']:08x} -> {out_png}")
    print(f"=== UART Output ===")
    print(emu.get_uart_output())
