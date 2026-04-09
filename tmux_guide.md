# tmux Guide

tmux is a terminal multiplexer — it lets you run persistent terminal sessions on a remote machine that keep going even after you disconnect. This is essential when working over SSH.

---

## Core Concepts

- **Session** — a persistent workspace that survives disconnects. You can have multiple sessions.
- **Window** — like a tab inside a session. Each window is a full terminal screen.
- **Pane** — a split within a window. Multiple terminals side by side.

---

## Installation

```bash
# macOS
brew install tmux

# Ubuntu / Debian
sudo apt install tmux

# Check version
tmux -V
```

---

## Basic Workflow

### Start a session
```bash
tmux new -s mysession
```
Give sessions meaningful names — you'll thank yourself later.

### Detach from a session (leave it running)
```
Ctrl+B, then D
```
You're dropped back to the normal shell. The session and everything in it keeps running.

### List active sessions
```bash
tmux ls
```

### Re-attach to a session
```bash
tmux attach -t mysession
```

### Kill a session
```bash
tmux kill-session -t mysession
```

---

## Essential Keybindings

All tmux shortcuts start with the **prefix**: `Ctrl+B`

| Keybinding | Action |
|---|---|
| `Ctrl+B, D` | Detach from session (session keeps running) |
| `Ctrl+B, C` | Create a new window |
| `Ctrl+B, N` | Next window |
| `Ctrl+B, P` | Previous window |
| `Ctrl+B, ,` | Rename current window |
| `Ctrl+B, %` | Split pane vertically |
| `Ctrl+B, "` | Split pane horizontally |
| `Ctrl+B, Arrow keys` | Move between panes |
| `Ctrl+B, Z` | Zoom current pane to full screen (toggle) |
| `Ctrl+B, X` | Kill current pane |
| `Ctrl+B, $` | Rename current session |
| `Ctrl+B, S` | List and switch between sessions |

---

## Typical SSH Use Case

```bash
# 1. SSH into your remote machine
ssh user@remote-machine

# 2. Start a tmux session
tmux new -s dev

# 3. Run your script or server
python train.py > results.log

# 4. Detach — your process keeps running
Ctrl+B, then D

# 5. Close your laptop, lose connection — doesn't matter

# 6. SSH back in later and re-attach
ssh user@remote-machine
tmux attach -t dev
# Your process is still running, output is intact
```

---

## Output Files

tmux has no effect on file I/O. Any files your script writes to disk are written normally — tmux only manages the terminal session, not the underlying process. Your output files will be exactly where you expect them.

To monitor a log file while detached, use:
```bash
tail -f results.log
```

---

## Tips

- **Always use named sessions** (`tmux new -s name`) rather than unnamed ones — easier to manage.
- **Scroll mode**: Press `Ctrl+B, [` to enter scroll mode and use arrow keys or Page Up/Down to browse output. Press `Q` to exit.
- **Multiple projects**: Use one session per project to keep things organised.
- **Already in tmux?**: Running `tmux` inside tmux is usually a mistake — check with `tmux ls` first and attach instead.
