# Raspberry Pi Zero 2 W — headless setup runbook

Bring-up notes for the rover-mounted Raspberry Pi Zero 2 W + CSI camera described in
[`circe_v1/docs/mothership-scout.md`](../../circe_v1/docs/mothership-scout.md). Written after a
setup session where **Raspberry Pi Imager's OS-customisation silently did nothing, twice** — this
documents the mechanism that actually works on the current image, and the evidence for why the
obvious routes don't.

Status: Pi is up, on WiFi, SSH + VNC reachable, camera capturing.

> **Secrets:** every password below is written as a placeholder. Do not commit the real WiFi PSK or
> account password to this repo.

## Confirmed hardware / image

| Thing | Value | How confirmed |
|---|---|---|
| Board | Raspberry Pi Zero 2 W | — |
| Camera | **IMX219** (Camera Module v2), 3280x2464 | `rpicam-hello --list-cameras` |
| OS | Raspberry Pi OS **Trixie / Debian 13**, pi-gen image `2026-06-18`, stage4 (labwc + lightdm) | `/etc/os-release`, `/etc/rpi-issue` |
| Kernel | `6.18.34+rpt-rpi-v8` aarch64 | login banner |
| libcamera | `v0.7.1+rpt20260609` | `rpicam-hello` output |
| Hostname / addr | `raspberrypi` / `raspberrypi.local` | mDNS via `avahi-daemon` (installed + enabled) |
| WiFi | 2.4 GHz, channel 1, WPA1/WPA2 | `nmcli device wifi list` |

The Zero 2 W is 2.4 GHz only. Verify the target AP is actually on 2.4 GHz before blaming anything
else — `nmcli -f SSID,CHAN,FREQ device wifi list` (channels 1–13 are 2.4 GHz).

## What does NOT configure this image

All three were tried and observed to no-op. Do not reach for them.

| Mechanism | Result | Evidence |
|---|---|---|
| **Raspberry Pi Imager** OS-customisation | Wrote **zero bytes** of config, across two separate flashes | Settings *were* saved in `~/.config/Raspberry Pi/Imager.conf`; the flash happened (`PARTUUID` changed); yet `find <boot-mount> -newermt <image build date>` returned **nothing**. Imager was `1.8.5+noembed-0ubuntu5` from the Ubuntu archive, against a 2026 image — suspected mismatch, **not proven**. |
| **`custom.toml`** | Not read at all by this image | `grep -rl 'custom\.toml'` across `/usr/lib`, `/usr/bin`, `/usr/sbin` on the rootfs returns nothing; there is no `firstboot` script. |
| **cloud-init** (`user-data` / `network-config`) | Present, enabled, ignored | `datasource_list: [NoCloud, None]` with `seedfrom: file:///boot/firmware`, six units enabled under `cloud-init.target.wants` — but a complete `user-data` was not applied on first boot (the setup wizard still ran). Suspected cause is the `instance_id` in `meta-data` (`rpios-image`) vs. the cached `/var/lib/cloud/data/instance-id` (`nocloud`); **not verified**. |

## What works: `firstrun.sh` + a `systemd.run=` hook

This is the mechanism Raspberry Pi Imager itself uses. Everything is written to the **FAT32 boot
partition**, which udisks mounts as your own user — **no `sudo` needed on the host machine.**

### 1. Generate a password hash

```bash
openssl passwd -6 'YOUR_ACCOUNT_PASSWORD'
```

### 2. Write `firstrun.sh` to the boot partition root

```bash
B=/media/$USER/bootfs          # adjust to your mount point
HASH='<paste the $6$... hash from step 1>'

cat > "$B/firstrun.sh" <<EOF
#!/bin/bash
set +e

/usr/lib/raspberrypi-sys-mods/imager_custom set_wlan 'YOUR_SSID' 'YOUR_WIFI_PSK' 'LK'
/usr/lib/raspberrypi-sys-mods/imager_custom enable_ssh -p
/usr/lib/userconf-pi/userconf 'pi' '$HASH'

rm -f /boot/firmware/firstrun.sh
sed -i 's| systemd.run.*||g' /boot/firmware/cmdline.txt
exit 0
EOF
chmod +x "$B/firstrun.sh"
```

### 3. Hook it into `cmdline.txt`

`cmdline.txt` **must remain a single line with no trailing newline** — append, never add a line.

```bash
sed -i '1 s|$| systemd.run=/boot/firmware/firstrun.sh systemd.run_success_action=reboot systemd.unit=kernel-command-line.target|' "$B/cmdline.txt"
```

### 4. Unmount, boot

```bash
sync && udisksctl unmount -b /dev/sdX1 && udisksctl unmount -b /dev/sdX2 && sync
```

First boot runs the script, then **reboots itself once** (`systemd.run_success_action=reboot`).
Allow ~2–3 minutes. The script deletes itself and strips its own hook afterwards.

### What each helper actually does

These are the image's own scripts — read them on the card rather than trusting forum snippets.
Source: `/usr/lib/raspberrypi-sys-mods/imager_custom`, `/usr/lib/userconf-pi/userconf`.

| Call | Effect |
|---|---|
| `imager_custom set_wlan SSID PASS COUNTRY` | Writes `/etc/NetworkManager/system-connections/preconfigured.nmconnection` (`type=wifi`, `key-mgmt=wpa-psk`, `psk=`, `[ipv4] method=auto`), `chmod 600`, and runs `raspi-config nonint do_wifi_country COUNTRY`. |
| `imager_custom enable_ssh -p` | `systemctl enable ssh` and sets `PasswordAuthentication yes` in `sshd_config`. `-k` instead gives key-only. Trailing args are treated as public-key lines appended to `~/.ssh/authorized_keys`. |
| `userconf NAME HASH` | Renames UID 1000 if needed, `usermod -s /bin/bash`, `chpasswd -e`, then **`cancel-rename`** — this is what suppresses the first-boot wizard and sets `autologin-user` in `/etc/lightdm/lightdm.conf`. |

`set_wlan` writes the PSK in plaintext into a root-owned `0600` file. That is stock Raspberry Pi
behaviour, but it means anyone with the SD card has the WiFi password.

## Verifying a card before you boot it

The single most useful diagnostic — settles "did anything customise this card?" in one command:

```bash
find /media/$USER/bootfs -newermt "<image build date>" -printf '%TY-%Tm-%Td %TH:%TM  %6s  %p\n'
```

A freshly-flashed, uncustomised card returns **nothing**. Get the build date from
`/etc/rpi-issue` on the rootfs.

Other quick checks (all readable without root — `system-connections/` is `0755`, only the files
inside are `0600`):

```bash
awk -F: '$3==1000' /media/$USER/rootfs/etc/passwd          # nologin shell => wizard has not run
cat /media/$USER/rootfs/etc/hostname
ls -la /media/$USER/rootfs/etc/NetworkManager/system-connections/
ls -la /media/$USER/rootfs/etc/ssh/ | grep host_key        # none => never booted
```

## Access

`~/.ssh/config` on the workstation:

```
Host pi
    HostName raspberrypi.local
    User pi
```

```bash
ssh pi
```

Boot-to-desktop is **already passwordless** — `userconf`'s `cancel-rename` sets `autologin-user=pi`
in `/etc/lightdm/lightdm.conf`, and `default.target -> graphical.target`. `sudo` and SSH still
prompt; an *empty* account password would not help, because `sshd` defaults to
`PermitEmptyPasswords no` and would lock you out. Use an SSH key if you want passwordless remote
login.

## VNC

Raspberry Pi OS ships **WayVNC**, not RealVNC — the official docs don't mention RealVNC at all.
RealVNC Connect 7.x/8.x wants a RealVNC account; skip it.

```bash
sudo raspi-config nonint do_vnc 0     # run this ON the Pi, with the desktop already up
sudo systemctl start wayvnc
```

**Timing matters.** `raspi-config`'s branch selector is:

```sh
is_labwc() { pgrep labwc > /dev/null; return $?; }
```

A *running-process* check. Run `do_vnc 0` early in boot (e.g. from a `firstrun.sh` or cloud-init
`runcmd`) and `pgrep labwc` fails, so it enables the useless RealVNC-X11 service instead of
`wayvnc.service`. Either run it once the desktop is up, or `systemctl enable wayvnc.service`
directly.

Client, from the workstation (TigerVNC — what the Raspberry Pi docs recommend, and it speaks the
RSA-AES auth WayVNC is configured for via `rsa_private_key_file` in `/etc/wayvnc/config`):

```bash
vncviewer raspberrypi.local:5900
```

Note `/etc/wayvnc/config` ships `address=::` — WayVNC listens on **all** interfaces, reachable by
anything on the LAN. `enable_auth=true` + `enable_pam=true` mean it demands the account password.
Bind to localhost and tunnel over SSH if that exposure isn't acceptable.

## Camera

```bash
rpicam-hello --list-cameras     # detection check
rpicam-hello -t 0               # live preview  -- needs a display: run inside VNC, not over SSH
rpicam-still -n -o ~/test.jpg   # headless capture, safe over SSH (-n = no preview)
rpicam-vid -t 10000 -o ~/test.h264
```

`rpicam-hello -t 0` over a plain SSH session has no compositor to draw on. Run it from a terminal
**inside** the VNC desktop, or use `-n`.

Streaming to the workstation instead of using VNC:

```bash
# on the Pi
rpicam-vid -t 0 --inline --listen -o tcp://0.0.0.0:8888

# on the workstation (needs mpv/ffplay/vlc installed)
mpv --profile=low-latency --untimed tcp://raspberrypi.local:8888
```

## USB gadget mode (not enabled — noted for later)

`rpi-usb-gadget` **1.0.6 is installed** on this image and the Zero 2 W is on its supported list. It
turns the micro-USB data port (**the one nearest the HDMI, not `PWR IN`**) into a USB Ethernet
device, giving SSH over a single cable to a laptop — no WiFi, no monitor, no keyboard. Useful if the
rover Pi ever needs bring-up without a network.

```bash
sudo rpi-usb-gadget on      # then reboot
sudo rpi-usb-gadget status
```

From `/usr/bin/rpi-usb-gadget`: writes `g_ether` to `/etc/modules-load.d/usb-gadget.conf`, appends
`dtoverlay=dwc2,dr_mode=peripheral` to `config.txt`, and creates two NetworkManager profiles —
`USB Gadget (client)` (DHCP, for host internet-sharing) and `USB Gadget (shared)`
(`ipv4.method shared` on `10.12.194.1/28`), with an ICS watcher service flipping between them.

Once enabled, that port is **networking + power only** — it stops working as a USB host port, so no
keyboard/mouse on it.

## Peripherals on the Zero 2 W

One micro-USB OTG data port (`USB`) and one power port (`PWR IN`), plus mini-HDMI. A mouse/keyboard
needs a micro-USB-male → USB-A-female OTG adapter, and a hub for more than one device. Bluetooth 4.2
+ BLE is onboard. The first-boot wizard is GTK and keyboard-navigable (Tab / arrows / Space /
Enter) if you have a keyboard but no mouse — though with this runbook the wizard never appears.

## References

- [Raspberry Pi — Remote access documentation](https://www.raspberrypi.com/documentation/computers/remote-access.html) (WayVNC is the shipped VNC server)
- [USB gadget mode in Raspberry Pi OS: SSH over USB](https://www.raspberrypi.com/news/usb-gadget-mode-in-raspberry-pi-os-ssh-over-usb/) (`rpi-usb-gadget` ships by default in Trixie images dated 2025-10-20 and later)
- [Raspberry Pi Zero 2 W product page](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) (port and radio specs)
- On-card sources of truth: `/usr/lib/raspberrypi-sys-mods/imager_custom`, `/usr/lib/userconf-pi/userconf`, `/usr/bin/rpi-usb-gadget`, `/usr/bin/raspi-config`
