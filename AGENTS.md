# Repository Guidelines

## Project Structure & Module Organization
- `main.cc`: demo pipeline entry; wires source, inference, tracking, OSD, broker, and destination nodes.
- `vp_node/`: core node framework and pipeline modules (`nodes/infer`, `nodes/osd`, `nodes/track`, `nodes/broker`).
- `videocodec/`: Rockchip-oriented decode/encode, demux/mux, and scaler components.
- `models/`: RKNN model wrappers and post-processing (`yolo*`, `rtmpose*`, `classifier*`).
- `tests/`: C++ sample test entry points (`main_model.cc`, `main_track.cc`, `main_record.cc`).
- `python/`: lightweight Python inference/tracking/count demos (`test_model.py`, `test_count.py`).
- `assets/`: configs, sample images/videos, and demo resources. Avoid committing large new media files unless required.

## Build, Test, and Development Commands
- `./build-linux.sh`: standard Linux build (creates `build/`, runs `cmake`, `make`, `make install`).
- `cmake -S . -B build && cmake --build build -j4`: explicit out-of-source build for iteration.
- `cmake --install build`: installs binaries/libs to `build/bin` per root `CMakeLists.txt`.
- `build/bin/rk_videopipe`: run the default C++ demo pipeline.
- `cd python && python3 test_model.py`: run image-based Python model smoke test.
- `cd python && python3 test_count.py`: run video counting/tracking smoke test.

## Coding Style & Naming Conventions
- C++17 is required; follow existing style: 4-space indentation, braces on new lines for functions, and readable include grouping.
- Use `snake_case` for variables/functions, and keep existing prefixes (`vp_`, `rk_`) for modules and classes.
- Keep node names and config paths explicit (example: `"rk_yolo_0"`, `"assets/configs/person.json"`).
- Edit dependency paths in `cmake/common.cmake` instead of hardcoding local paths in source files.

## Testing Guidelines
- No unified `ctest`/`gtest` suite is configured; use smoke tests from `tests/` and `python/test_*.py`.
- Name new Python checks as `python/test_<feature>.py`.
- For C++ changes, verify at least: build success, pipeline startup, and one sample media run from `assets/`.

## Commit & Pull Request Guidelines
- Follow current history style with short imperative subjects: `Add ...`, `Fix ...`, `Update ...`.
- Keep subject lines concise and scoped to one change area.
- PRs should include: purpose, affected modules (for example `vp_node/nodes/infer`), validation steps/commands, and runtime evidence (logs or screenshots for OSD/streaming changes).
- Link related issues and clearly mark dependency or environment assumptions (RK3588 board, FFmpeg/GStreamer plugin status).

## Security & Configuration Tips
- Do not commit secrets (RTSP credentials, tokens, private endpoints) in code or config.
- Verify third-party symlinks (`mpp`, `rknn-toolkit2`) and runtime libraries before running on new devices.
