# 3DCV 2026 竞赛

面向 3DCV 2026 竞赛的桌面物体识别与计数框架。系统从 RGB-D 帧源（离线图片序列或 OpenNI 实时相机）取帧，经过一条固定的处理流水线（检测 → 桌面定位 → 深度过滤 → OCR → 计数 → 可视化），由轮次状态机编排整场比赛的流程，并把最终识别结果写成文件、通过 TCP 发送给裁判盒。

整个项目采用「**组件 + 工厂 + 配置**」的可插拔架构：每类能力都有抽象基类和若干实现，由对应的 `build_*` 工厂按配置里的 `type` 字段选择实现；顶层配置通过 `!include` 把各领域配置拼装起来。这样更换检测器、帧源、裁判通信方式等，基本只需要改配置。

***

## 目录

- [环境与依赖](#环境与依赖)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [整体架构](#整体架构)
- [处理流水线（Pipeline）](#处理流水线pipeline)
- [轮次状态机（State Machine）](#轮次状态机state-machine)
- [结果文件格式](#结果文件格式)
- [配置系统总览](#配置系统总览)
- [配置文件详解](#配置文件详解)
  - [顶层配置 config.yaml](#顶层配置-configyaml)
  - [state\_machine 状态机](#state_machine-状态机)
  - [frame\_source 帧源](#frame_source-帧源)
  - [referee 裁判通信](#referee-裁判通信)
  - [class\_registry 类别注册表](#class_registry-类别注册表)
  - [detector 检测器](#detector-检测器)
  - [table\_locator 桌面定位器](#table_locator-桌面定位器)
  - [filter 深度过滤器](#filter-深度过滤器)
  - [ocr 文字识别](#ocr-文字识别)
  - [counter 计数器](#counter-计数器)
- [环境变量](#环境变量)
- [日志与产物](#日志与产物)

***

## 环境与依赖

- Python 3.10
- 依赖见 [requirements.txt](requirements.txt)：

| 依赖              | 用途                       |
| --------------- | ------------------------ |
| `PyYAML`        | 加载 YAML 配置，支持 `!include` |
| `numpy`         | 张量/数值计算（锁定 `<2`）         |
| `opencv-python` | 图像处理与可视化窗口               |
| `onnxruntime`   | ONNX 推理后端（通用平台）          |
| `open3d`        | 深度过滤器的可选点云可视化            |
| `openni`        | OpenNI 实时相机取帧            |
| `Pillow`        | 图像辅助处理（OCR 引擎依赖）          |
| `shapely`       | OCR 文本检测框的多边形运算（DB 后处理）   |
| `pyclipper`     | OCR 文本检测框的多边形扩张（DB 后处理）   |
| `rapidfuzz`     | OCR 文本与模板的模糊匹配            |

安装：

```bash
python3 -m pip install -r requirements.txt
```

> 香橙派（Orange Pi AiPro）等昇腾平台使用 ACL（`.om` 模型）推理，需自带昇腾 CANN/AclLite 运行时，相关封装位于 `core/infra/inference/acl/`，不通过 pip 安装。

***

## 快速开始

```bash
# 第一轮（单桌识别）
python3 main.py --round round1 --config config/config.yaml

# 第二轮（多桌轮转）
python3 main.py --round round2 --config config/config.yaml
```

也可以直接用封装好的脚本：

```bash
./start_r1.sh   # 等价于 --round round1
./start_r2.sh   # 等价于 --round round2
```

命令行参数（见 [main.py](main.py)）：

| 参数         | 是否必填 | 说明                              |
| ---------- | ---- | ------------------------------- |
| `--round`  | 是    | `round1` 或 `round2`，选择要运行的轮次    |
| `--config` | 否    | 运行时配置路径，默认 `config/config.yaml` |

进程退出码：`0` 正常结束；`130` 被中断（Ctrl+C / SIGTERM）；`1` 运行失败。收到 `SIGINT`/`SIGTERM` 时会统一转成 `KeyboardInterrupt`，先尽力保存并上报已有结果，再退出。

***

## 目录结构

```
3dcv/
├── main.py                     # 命令行入口，解析参数、注册信号、调用 run_round
├── start_r1.sh / start_r2.sh   # 两轮的启动脚本
├── requirements.txt
├── config/                     # 所有 YAML 配置（见下文配置详解）
├── models/                     # 权重（YOLO .onnx/.om、OCR ppocrv5/）
├── runs/                       # 运行日志输出目录（logging.base_dir）
├── result/                     # 结果文件示例目录
├── scripts/                    # 相机标定、D2C 对齐等独立调试脚本
└── core/
    ├── app.py                  # 运行时装配：按配置构建所有组件并交给状态机
    ├── config_loader.py        # YAML 加载 + !include mixin 合并
    ├── types.py                # Frame / Detection / RecognitionItem 等数据结构
    ├── utils/platform.py       # 平台探测（macos / linux / orangepi …）
    ├── orchestration/          # 编排层
    │   ├── pipeline/           # 逐帧处理流水线 FramePipeline
    │   └── state_machine/      # 轮次状态机、帧采集器、状态日志器
    ├── components/             # 可插拔业务组件（每类含 base + 实现 + builder）
    │   ├── frame_source/       # 帧源：图片序列 / OpenNI 相机
    │   ├── detector/           # 检测器：按版本隔离（如 yolov11/）
    │   ├── table_locator/      # 桌面定位器
    │   ├── filter/             # 深度过滤器
    │   ├── ocr/                # OCR：PaddleOCR ONNX 引擎 + 书本文字分类
    │   ├── counter/            # 计数器：贝叶斯计数
    │   └── referee/            # 裁判通信客户端（TCP socket）
    └── infra/                  # 基础设施
        ├── inference/          # 推理后端：onnx / acl
        ├── logging/            # 结构化事件日志
        └── visualization/      # OpenCV 可视化 / 空实现
```

***

## 整体架构

一次运行的调用链（[core/app.py](core/app.py) 的 `run_round`）：

```
main.py
  └─ run_round(config_path, round_name)
       ├─ load_config()                     # 加载并合并配置
       ├─ EventLogger(...)                   # 建立日志会话
       ├─ 依次构建组件（记录每个组件的初始化耗时）：
       │     frame_source, referee_client, detector,
       │     table_locator, filter, ocr, counter, visualizer
       ├─ FramePipeline(...)                 # 把上述组件注入统一流水线
       └─ Round1StateMachine / Round2StateMachine(...).run()
```

关键分工：

- **工厂（builder / factory）**：每个组件目录下的 `build_*` 函数按配置 `type` 选择实现，实现之间通过抽象基类解耦。
- **FramePipeline**：只负责「处理一帧」的固定顺序，不关心比赛节奏。
- **StateMachine**：只负责「比赛节奏」（何时开始、每桌处理多久、何时换桌、何时提交结果），不关心单帧细节。
- **EventLogger**：贯穿全程，把结构化事件（组件初始化、状态进入/退出、每桌提交、异常等）写入日志。

***

## 处理流水线（Pipeline）

实现见 [core/orchestration/pipeline/frame\_pipeline.py](core/orchestration/pipeline/frame_pipeline.py)。`process_frame(frame, table)` 对单帧依次执行：

1. **检测（detect）** — 检测器对 RGB 图推理，得到图像坐标下的检测框。
2. **桌面定位（locate）** — 桌面定位器跟踪「Table」检测，稳定后锁定桌面框。**若尚未定位成功，直接返回当前计数**（此时过滤/计数不可靠，跳过后续步骤）。
3. **深度过滤（filter）** — 借助深度图把检测投影到 3D，只保留落在桌面范围内的目标。
4. **OCR** — 对候选目标（默认 `Book`）裁剪后做文字识别，把识别文本模糊匹配到书本物品名，命中则作为额外检测项合并进来。
5. **计数（counter update）** — 剔除 `ignored_by_counter` 中的类别后，用当前帧更新计数器。
6. **可视化（visualize）** — 每个阶段都会调用可视化器渲染（仅渲染配置里开启的 `stages`）。

另外两个入口：

- `track_frame(frame, table)`：只跑「检测 + 桌面定位」，用于第二轮开始识别前的**桌面锁定阶段**。
- `preview_frame(frame, table)`：只渲染，不改变任何识别状态。

`get_items(table)` 把计数器当前状态转成提交用的 `RecognitionItem` 列表；`reset_table_state()` 在换桌/换窗口时清空桌面定位器和计数器状态。

***

## 轮次状态机（State Machine）

状态机负责整场比赛的编排。基类 [BaseStateMachine](core/orchestration/state_machine/base.py) 提供两个公共能力：

- `_write_and_send_result(items, reason, strict)`：先把结果**原子落盘**（写临时文件再替换），再按需通过裁判客户端发送；`strict=True` 时发送失败会抛异常。
- `_close_runtime_resources()`：按 `pipeline → referee_client → frame_source` 顺序关闭资源，保证只执行一次、单个失败不影响其它。

帧的采集由 [FrameCollector](core/orchestration/state_machine/frame_collector.py) 负责：按「名义时长 × `time_scale`」换算出实际采集时长，在时间窗内逐帧产出（至少产出一帧）。每个状态的进入/退出与耗时由 [StateLogger](core/orchestration/state_machine/state_logger.py) 记录。

### 第一轮（Round1）

见 [round1\_state\_machine.py](core/orchestration/state_machine/round1_state_machine.py)。只处理 1 号桌：

```
INIT
CONNECT_REFEREE          # 连接裁判盒（失败即报错）
SEND_START               # 发送开始信号
INIT_PIPELINE            # 重置流水线状态
ACQUIRE_TABLE            # 在 max_acquire_sec 内尝试稳定锁定桌面；
                         #   超时回退到 default_bbox
PROCESS_TABLE_1_WINDOW   # 在 round1_recognize_sec 时间窗内逐帧识别，
                         #   每帧把最新结果落盘（便于中断兜底）
COMMIT_TABLE_1           # 提交 1 号桌结果
SAVE_RESULT / SEND_RESULT  # 写结果并上报（strict=True）
STOP                     # 关闭资源
```

### 第二轮（Round2）

见 [round2\_state\_machine.py](core/orchestration/state_machine/round2_state_machine.py)。按 `round2_table_durations_sec` 依次处理多张桌子，每桌四步：

```
INIT → CONNECT_REFEREE → SEND_START → INIT_PIPELINE

for 每张桌子:
  1) ACQUIRE_TABLE            # 在 max_acquire_sec 内尝试稳定锁定桌面；
                             #   超时回退到 default_bbox
  2) PROCESS_TABLE_N_WINDOW  # 在该桌 duration 内逐帧识别，持续落盘最新结果
  3) COMMIT_TABLE_N          # 提交当前桌结果，并入总结果
  4) 若还有下一桌:
       ROTATE                # 通知裁判旋转/换桌
       NEXT_SEQUENCE         # 切换帧源到下一桌对应序列
       RESET_TABLE_STATE     # 重置流水线状态

SAVE_RESULT / SEND_RESULT → STOP
```

### 异常与中断处理

两轮的 `run()` 都用 `try/except BaseException/finally` 包裹：任何异常（包括 Ctrl+C）都会先调用 `_finalize_interrupted_run()`，以 `reason="interrupted"`、`strict=False` 的方式**尽力保存并上报当前结果**（第二轮会把已提交结果 + 进行中桌位的当前计数合并），然后再向上抛出；`finally` 中统一关闭资源。

***

## 结果文件格式

由 [RefereeSocketClient.render\_result](core/components/referee/socket_client.py) 生成，落盘文件名为 `{file_prefix}-R{轮次}.txt`（如 `USTB-MQMM5-R2.txt`）。数量为 0 的条目会被跳过：

```
START
Goal_ID=CA001;Num=2;Table=1
Goal_ID=CB003;Num=1;Table=2
END
```

裁判通信采用大端头部 `>ii`（数据类型 + 负载长度）+ UTF-8 负载的协议，数据类型：`0` 开始信号、`1` 结果文件、`3` 旋转/换桌。

***

## 配置系统总览

配置加载见 [core/config\_loader.py](core/config_loader.py)：

- 支持自定义 `!include path.yaml` 标签，把其它 YAML 文件的内容内联进来（相对当前文件解析路径）。
- 顶层的 `mixins` 是一个列表，每个元素必须是一个「**只含单一顶层域**」的映射（通常就是一个 `!include`）。加载时会把每个 mixin 合并到根配置，且**各 mixin 的顶层域不能重复**，否则报错。

也就是说，[config.yaml](config/config.yaml) 是「主装配文件」，通过 `mixins` 组合出 `state_machine` / `frame_source` / `referee` / `class_registry` / `detector` / `table_locator` / `filter` / `ocr` / `counter` 等各领域配置。想切换某个领域的实现（如把裁判从 disabled 换成 default、把帧源从 eval\_videos 换成 openni\_astra），改对应的 `!include` 行即可。

***

## 配置文件详解

> 说明：下面标注的「默认值」指**代码里** **`config.get(key, 默认值)`** **的兜底值**；仓库自带的 YAML 通常已显式给出取值。

### 顶层配置 config.yaml

[config/config.yaml](config/config.yaml) 直接定义 `team` / `logging` / `visualization` 三个域，并用 `mixins` 引入其余领域。

**`team`** — 队伍信息：

| 键             | 说明                                           |
| ------------- | -------------------------------------------- |
| `file_prefix` | 结果文件名前缀，如 `USTB-MQMM5` → `USTB-MQMM5-R1.txt` |
| `team_id`     | 队伍 ID，随开始信号发送给裁判盒（示例为 `REPLACE-HERE`，需替换）    |

**`logging`** — 日志（对应 [EventLogger](core/infra/logging/event_logger.py)）：

| 键           | 默认      | 说明                                        |
| ----------- | ------- | ----------------------------------------- |
| `base_dir`  | —       | 日志输出目录，文件名形如 `20260805_101530_round2.log` |
| `console`   | `true`  | 是否同时输出到控制台                                |
| `per_frame` | `false` | 是否记录每帧的流水线细节事件（调试用，日志量大）                  |

**`visualization`** — 可视化（对应 [OpenCvVisualizer](core/infra/visualization/opencv_visualizer.py)）：

| 键                     | 默认                        | 说明                                                                                              |
| --------------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| `enabled`             | `false`                   | 是否启用可视化；关闭时用空实现（不开窗口）                                                                           |
| `type`                | —                         | 可视化器类型，目前仅 `opencv`                                                                             |
| `mode`                | `rgb_depth`               | 显示模式：`rgb` / `depth` / `rgb_depth`（并排）/ `rgb_depth_overlay`（深度伪彩叠加到 RGB）                        |
| `window_name`         | `3DCV Debug View`         | 窗口标题                                                                                            |
| `depth_max_mm`        | `5000`                    | 深度伪彩色化的最大量程（毫米），超出会被截断                                                                          |
| `depth_overlay_alpha` | `0.45`                    | overlay 模式下深度层的透明度（0\~1）                                                                        |
| `wait_key_ms`         | `1`                       | 每帧 `cv2.waitKey` 的毫秒数                                                                           |
| `draw_labels`         | `true`                    | 是否在检测框上标注类别与置信度                                                                                 |
| `draw_status_label`   | `true`                    | 是否绘制状态信息文字（如 FPS 等）                                                                             |
| `draw_state_label`    | `true`                    | 是否绘制当前状态机状态名                                                                                    |
| `overlay_text_scale`  | `0.45`                    | 叠加文字的字号缩放                                                                                       |
| `overlay_line_height` | `18`                      | 叠加文字的行高（像素）                                                                                     |
| `fps_smoothing`       | `0.8`                     | FPS 显示的指数平滑系数（0\~0.99）                                                                          |
| `stages`              | `[preview, track, final]` | 需要渲染的阶段列表，可选：`preview` / `detect` / `track` / `locate` / `filter` / `ocr` / `final`。开启多个阶段会分别开窗 |

***

### state\_machine 状态机

[config/state\_machine/default.yaml](config/state_machine/default.yaml)，对应两轮状态机与 [FrameCollector](core/orchestration/state_machine/frame_collector.py)。

| 键                            | 示例                   | 说明                                                      |
| ---------------------------- | -------------------- | ------------------------------------------------------- |
| `time_scale`                 | `1.0`                | 时间缩放系数。所有名义时长乘以它得到实际采集时长；离线回放时可调大/调小以加速或减速              |
| `settle_wait_sec`            | `2.0`                | 锁定桌面前的最短稳定等待时长（两轮均生效）；锁定阶段至少等这么久再判断是否锁定成功              |
| `max_acquire_sec`            | `5.0`                | 锁定桌面的最长等待时长（两轮均生效）；超时则回退到 `table_locator.default_bbox` |
| `round1_recognize_sec`       | `28.0`               | 第一轮单桌识别的时间窗（秒）                                          |
| `round2_table_durations_sec` | `[15.0, 15.0, 40.0]` | 第二轮各桌识别时长（秒），**列表长度即桌子数量**                              |

***

### frame\_source 帧源

帧源有两种实现，通过不同的 `!include` 切换。工厂见 [frame\_source/builder.py](core/components/frame_source/builder.py)：先按当前平台把 `base_paths` 解析成单个 `base_path`，再把 `common` 与该轮次专属配置合并。

#### A. 离线图片序列 — [config/frame\_source/eval\_videos.yaml](config/frame_source/eval_videos.yaml)

用于回放录制好的评测数据。对应 [ImageSequenceFrameSource](core/components/frame_source/image_sequence_reader.py)。

顶层：

| 键            | 说明                                                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `type`       | `image_sequence`                                                                                                       |
| `base_paths` | 平台 → 数据根目录 的映射。按 `current_platform()` 选择（如 `macos` / `orangepi`），找不到则回退 `default`，再回退 `base_path`。平台名会做归一化（下划线转连字符、小写） |
| `common`     | 所有轮次共用的读取参数                                                                                                            |
| `rounds`     | 各轮次专属参数（`round1` / `round2`）                                                                                           |

`common` 内：

| 键                             | 默认                    | 说明                                |
| ----------------------------- | --------------------- | --------------------------------- |
| `read_interval`               | `1`                   | 每隔几帧取一帧（必须为正）                     |
| `fps`                         | `0.0`                 | 回放帧率节流；`>0` 时按此帧率控制取帧速度，`0` 表示不节流 |
| `loop`                        | `false`               | 全部序列播放完是否循环                       |
| `rgb_dir` / `depth_dir`       | `rgb` / `depth`       | 每段序列内 RGB / 深度子目录名                |
| `rgb_suffix` / `depth_suffix` | `_rgb` / `_depth`     | 从文件名提取 `frame_id` 时要去掉的后缀         |
| `rgb_extensions`              | `[.jpg, .jpeg, .png]` | RGB 图允许的扩展名                       |
| `depth_extensions`            | `[.png]`              | 深度图允许的扩展名                         |
| `convert_rgb`                 | `true`                | 读入后是否做 BGR→RGB 转换                 |

`rounds.<round>` 内：

| 键            | 说明                                            |
| ------------ | --------------------------------------------- |
| `sequences`  | 该轮次要播放的序列子目录名列表（按序播放）。Round1 通常一段，Round2 每桌一段 |
| `transition` | （可选，多用于 Round2）序列间滑动转场动画配置                    |

`transition` 子项：

| 键                       | 可选值 / 默认                    | 说明                         |
| ----------------------- | --------------------------- | -------------------------- |
| `enabled`               | `false`                     | 是否启用序列间滑动转场                |
| `direction`             | `left` / `right` / `random` | 滑动方向；`random` 时随机选一个方向     |
| `duration_frames_range` | `[min, max]`                | 转场持续帧数的随机区间（`0<=min<=max`） |
| `random_seed`           | `null`                      | 随机种子；`null` 表示不固定          |
| `depth_mode`            | `nearest`                   | 转场时深度合成方式，目前仅支持 `nearest`  |

#### B. OpenNI 实时相机 — [config/frame\_source/openni\_astra.yaml](config/frame_source/openni_astra.yaml)

用于 Astra 类深度相机实时取帧。对应 [OpenNIFrameSource](core/components/frame_source/openni_reader.py)。

| 键                    | 示例            | 说明                                                                          |
| -------------------- | ------------- | --------------------------------------------------------------------------- |
| `type`               | `openni`      | —                                                                           |
| `openni_lib`         | `null`        | OpenNI2 库路径；`null` 时按环境变量与常见目录自动探测                                          |
| `width` / `height`   | `640` / `480` | 采集分辨率                                                                       |
| `fps`                | `30`          | 采集帧率                                                                        |
| `color_source`       | `auto`        | 彩色来源：`auto`（优先 OpenNI，退回 UVC）/ `openni` / `uvc` / `none`                    |
| `uvc_device`         | `/dev/video0` | 使用 UVC 彩色时的设备号/路径                                                           |
| `allow_unregistered` | `true`        | 硬件 D2C 不支持时是否允许继续（不报错）                                                      |
| `mirror`             | `false`       | 是否镜像画面                                                                      |
| `d2c.mode`           | `hardware`    | 深度到彩色对齐模式：`hardware`（相机内部 registration）/ `off`（原始深度不对齐） |

实时相机模式下，`OpenNIFrameSource` 会从 Orbbec 设备读取当前相机内参并传给 `DepthFilter`，覆盖 [config/filter/depth\_filter.yaml](config/filter/depth_filter.yaml) 里的静态内参。若 `d2c.mode=hardware` 且 registration 成功，过滤器使用 `color` 内参；若关闭 D2C 或 registration 不可用，则使用 `depth` 内参。离线图片/视频评估没有真实设备，因此继续使用配置文件里的内参。

***

### referee 裁判通信

两个预设：[default.yaml](config/referee/default.yaml)（启用）与 [disabled.yaml](config/referee/disabled.yaml)（关闭，本地调试用）。对应 [RefereeSocketClient](core/components/referee/socket_client.py)。

| 键                     | 示例                 | 说明                                    |
| --------------------- | ------------------ | ------------------------------------- |
| `enabled`             | `true` / `false`   | 是否真正联网。`false` 为「空跑」：仍写结果文件，但不连接、不发送  |
| `ip`                  | `192.168.1.88`     | 裁判盒 IP                                |
| `port`                | `6666`             | 裁判盒端口                                 |
| `result_base_dir`     | `~/Desktop/result` | 结果文件落盘目录（可被环境变量 `3DCV_RESULT_DIR` 覆盖） |
| `connect_retry_sec`   | `1.0`              | 连接失败后的重试间隔（秒）                         |
| `connect_timeout_sec` | `3.0`              | 连接总超时（秒），超时未连上则视为失败                   |

***

### class\_registry 类别注册表

[config/class\_registry/default.yaml](config/class_registry/default.yaml)。定义检测器输出 ID 与业务类别的映射，以及各类别在流水线中的角色。被检测器、计数器共享。

| 键                       | 说明                                      |
| ----------------------- | --------------------------------------- |
| `result_classes`        | Pipeline 内部需要计数的物品名称全集，不允许重复  |
| `result_class_to_goal_id` | 物品名称 → 比赛 `Goal_ID` 的映射，仅最终写结果时使用 |
| `internal_classes`      | 仅内部使用、不直接计入结果的类别（如 `Table`、`Book`）      |
| `detector_id_to_class`  | 检测器类别 ID → 类别名 的映射（模型输出的整数 ID 如何翻译成类别名） |
| `ignored_by_counter`    | 计数阶段要忽略的类别（如 `Table`、`Book` 不计入数量）      |
| `ocr_candidate_classes` | 作为 OCR 输入候选的类别（如 `Book`）                |
| `ocr_output_classes`    | OCR 可能产出的书本物品名称            |
| `ocr_templates`         | OCR 输出类别 → 模板文本 的映射，用于把识别文本模糊匹配成类别 |
| `bbox_colors`           | 物品名称 → RGB bbox 颜色；OpenCV 可视化时自动转为 BGR |

> 计数器把类别分为两类：一般类别用贝叶斯后验估计数量；`ocr_output_classes`（即「未知/难建模」类别）退化为取观测窗口内的最大值。

***

### detector 检测器

[config/detector/yolo/yolov11.yaml](config/detector/yolo/yolov11.yaml)。对应 [YOLOv11Detector](core/components/detector/yolov11/detector.py)。YOLO 不同版本的预处理、输出张量和后处理容易分叉，因此代码按版本隔离在 `core/components/detector/yolov11/` 这类目录下。

| 键                              | 示例                                                | 说明                                                                                                         |
| ------------------------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `type`                         | `yolov11`                                         | 检测器类型（决定使用哪个版本目录下的实现）                                                                                           |
| `backend`                      | `auto`                                            | 推理后端：`auto` / `onnx` / `acl`。`auto` 时香橙派选 `acl`（`.om`），其它平台选 `onnx`（`.onnx`）。可被环境变量 `3DCV_YOLO_BACKEND` 覆盖 |
| `weights`                      | `models/yolov11_...`                              | 权重路径前缀，**不含扩展名**；扩展名由后端自动补齐（`.om` / `.onnx`）                                                               |
| `input_width` / `input_height` | `640` / `640`                                     | 网络输入尺寸（letterbox 目标尺寸）                                                                                     |
| `conf_thresh`                  | `0.1`                                             | 置信度阈值，低于则丢弃                                                                                                |
| `nms_thresh`                   | `0.7`                                             | NMS 的 IoU 阈值                                                                                               |
| `graph_optimization_level`     | `ORT_ENABLE_ALL`                                  | ONNX Runtime 图优化级别（onnx 后端）                                                                                |
| `providers`                    | `[CoreMLExecutionProvider, CPUExecutionProvider]` | ONNX Runtime 的 execution provider 优先级列表                                                                    |

RGBD 版本使用 [config/detector/yolo/yolov11_rgbd.yaml](config/detector/yolo/yolov11_rgbd.yaml)，
对应 [YOLOv11RgbdDetector](core/components/detector/yolov11_rgbd/detector.py)。它与原
`yolov11` detector 完全隔离，但后处理语义对齐 Ultralytics：`xywh2xyxy`、按类别偏移的
NMS、`max_det`、`agnostic_nms`、`scale_boxes` 坐标还原。启用方式是把
[config.yaml](config/config.yaml) 中 detector include 替换为：

```yaml
- !include detector/yolo/yolov11_rgbd.yaml
```

RGBD detector 额外配置：

| 键             | 示例   | 说明 |
| -------------- | ------ | ---- |
| `depth_max_mm` | `3000` | 深度毫米图裁剪归一化上限，推理输入第 4 通道为 `clip(depth, 0, depth_max_mm) / depth_max_mm` |
| `pad_value`    | `114`  | RGBD 四通道 letterbox 填充值，默认与 Ultralytics 训练端多通道 `LetterBox` 保持一致 |
| `max_det`      | `300`  | NMS 后最多保留检测框数 |
| `max_nms`      | `30000` | 进入 NMS 前最多候选数 |
| `max_wh`       | `7680` | 类别偏移 NMS 使用的最大宽高常量 |
| `agnostic_nms` | `false` | 是否做类别无关 NMS |

***

### table\_locator 桌面定位器

[config/table\_locator/default.yaml](config/table_locator/default.yaml)。对应 [TableLocator](core/components/table_locator/table_locator.py)，通过多候选跟踪 + 稳定性打分锁定桌面框。

| 键                       | 示例                  | 说明                              |
| ----------------------- | ------------------- | ------------------------------- |
| `type`                  | `table_locator`     | 实现类型                            |
| `init_frames`           | `10`                | 定位初始化观察帧数：累计到该帧数后才尝试从候选中确认桌面    |
| `iou_threshold`         | `0.5`               | 把新检测关联到已有候选轨迹的 IoU 阈值           |
| `min_area`              | `400`               | 有效桌面框的最小面积（像素²），过小视为无效          |
| `update_interval`       | `5`                 | 锁定后每隔多少帧才尝试微调一次目标框              |
| `update_iou_threshold`  | `0.5`               | 微调目标框时，新检测需达到的 IoU 阈值           |
| `default_bbox`          | `[150,100,490,380]` | 默认桌面框 `[x1,y1,x2,y2]`；锁定超时时回退到它 |
| `min_stable_frames`     | `8`                 | 判定「稳定」所需的连续帧数                   |
| `max_center_shift_px`   | `8`                 | 相邻帧中心位移上限（像素），超过视为不稳定           |
| `max_size_change_ratio` | `0.05`              | 尺寸相对平均值的最大波动比例                  |
| `max_area_trend_ratio`  | `0.08`              | 面积首尾变化相对平均值的最大比例                |
| `max_missing_frames`    | `5`                 | 候选连续丢失超过该帧数则被清除                 |

***

### filter 深度过滤器

[config/filter/depth\_filter.yaml](config/filter/depth_filter.yaml)。对应 [DepthFilter](core/components/filter/depth_filter.py)：用相机内参把检测反投影到 3D，只保留落在桌面范围内的目标。实时相机运行时优先使用帧源从设备读取到的内参；离线评估使用本配置中的 `intrinsic`。

**`intrinsic`** — 离线评估内参（用于反投影）：

| 键                  | 说明       |
| ------------------ | -------- |
| `width` / `height` | 内参对应的分辨率 |
| `fx` / `fy`        | 焦距（像素）   |
| `cx` / `cy`        | 主点坐标     |

当前配置值来自 [runs/orbbec\_camera\_params.yaml](runs/orbbec_camera_params.yaml) 的 `color` 内参，适配采集评估视频时已经对齐到 RGB 坐标系的深度图。

**`depth_filter`** — 过滤参数，按用途分组：

`object_depth`：物品深度估计与空洞兜底。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `coord_sample_num` | `20` | 框内有效深度像素不足时的随机采样点数 |
| `min_valid_pixels` | `5` | 框内至少多少个有效深度像素才直接采用框内中位深度 |
| `expand_ratios` | `[0.15, 0.3]` | 框内深度不足时，按比例外扩 bbox 到邻域查找有效深度 |
| `table_plane_fallback` | `true` | 邻域仍无深度时，是否用 bbox 中心射线与桌面平面交点兜底 |
| `plane_offset_m` | `0.0` | 桌面平面交点的 table-y 偏移量；`0` 表示落在桌面平面上 |

`filtering`：过滤输出策略。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `keep_table` | `true` | 过滤后是否保留桌子（`Table`）本身 |
| `use_support_point` | `true` | 非桌面物体优先用 bbox 底部中心与桌面平面的交点做 footprint 判断 |

`visualization`：Open3D 点云窗口。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否开启 open3d 点云可视化（需要 `open3d`） |
| `window.width` / `window.height` | `640` / `480` | 可视化窗口尺寸 |
| `depth_trunc_m` | `2.0` | 点云可视化时的深度截断距离（米） |
| `point_cloud_stride` | `2` | 点云可视化的采样步长（越大越稀疏、越快） |
| `show_detection_boxes` | `true` | Open3D 点云窗口中是否显示检测 2D 框的 3D 投影线框 |
| `detection_box_depth_mode` | `median` | 框投影深度来源：`median` / `center` |
| `table_box_height_m` | `0.06` | Open3D 绿色桌面过滤框的显示高度（米） |

`plane_fit`：桌面平面拟合。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否优先用 `Table` 框内深度点拟合桌面平面 |
| `sample_stride` | `4` | 桌面平面拟合采样步长，越小点越多、越慢 |
| `max_points` | `3000` | 平面拟合最多使用的点数 |
| `min_points` | `80` | 少于该点数则认为平面拟合不可用 |
| `ransac_iterations` | `80` | RANSAC 平面拟合迭代次数 |
| `update_interval_frames` | `2` | 平面模型每隔多少帧重新拟合一次，中间复用稳定模型 |
| `early_stop_inlier_ratio` | `0.75` | RANSAC 内点比例达到该值时提前停止 |
| `distance_thresh_m` | `0.015` | 点到桌面平面的内点距离阈值（米） |
| `min_inlier_ratio` | `0.35` | 桌面平面内点比例低于该值则认为拟合失败 |
| `use_unmasked_points_for_orientation` | `true` | 平面法向用 mask 后点拟合，水平朝向/范围优先用未 mask 且贴近平面的桌面点估计 |
| `roi_shrink_ratio` | `0.04` | 桌子框采样前向内收缩比例，减少框外背景干扰 |

`footprint`：桌面范围与圆/方桌判断。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `mode` | `by_table` | 桌面 footprint 类型：`by_table` / `auto` / `rectangle` / `ellipse` |
| `table_modes` | `{1: rectangle, 2: rectangle, 3: ellipse}` | `by_table` 时按桌号指定 footprint；第一轮 table=1，因此也是方桌 |
| `range_percentile` | `2.0` | 估计桌面边界时丢弃两端离群点的百分比 |
| `size_scale` | `1.0` | 平面拟合成功时，对自动估计桌面长宽做等比例缩放 |
| `range_margin_m` | `0.05` | 自动估计桌面范围后额外扩出的边缘余量（米） |
| `vertical_expansion.enabled` | `true` | 是否按物体 table-y 高度把 footprint 向外扩成倒置梯形/锥台 |
| `vertical_expansion.angle_deg` | `35.0` | 相对 table-y 轴的张开角度；越大，物体越高时 `x/z` 允许范围扩得越快 |
| `vertical_expansion.max_extra_margin_m` | `0.05` | 高度扩张带来的额外边缘余量上限（米），防止远离桌面的物体被放进来 |
| `vertical_expansion.min_height_m` | `0.0` | 起扩高度（米）；物体高出 `y_min` 超过该值后才开始按角度扩张 |
| `round_fill_ratio_max` | `0.88` | `auto` 时凸包面积/外接矩形面积低于该值则按圆/椭圆桌处理 |
| `round_corner_ratio_max` | `0.03` | `auto` 时外接矩形角落占用率低于该值则按圆/椭圆桌处理 |
| `switch_min_frames` | `3` | `auto` 模式下连续多少帧判断为另一类型才切换 footprint |

`vertical_expansion` 的额外扩张量按如下方式计算：

```text
extra_margin = tan(angle_deg) * max(0, center_y - y_min - min_height_m)
extra_margin = min(extra_margin, max_extra_margin_m)
```

其中 `center_y` 是检测点在桌面坐标系中的 table-y 高度。它只改变 footprint 的
`x/z` 判断范围，不改变 `above_y` 高度门槛；用途是让桌面上方的立体物体在边缘处
有更合理的容差，同时避免像直接增大 `range_margin_m` 那样把桌面平面附近的地面/桌外
物品整体放进来。

`smoothing`：桌面模型时间平滑与离群拒绝。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `alpha` | `0.25` | 桌面坐标系与绿框的时间平滑系数；越小越稳但响应越慢 |
| `reset_distance_m` | `0.35` | 桌面原点跳变超过该距离时重置平滑状态（米） |
| `reject_distance_m` | `0.12` | 相对上一帧稳定桌面原点跳变超过该距离时拒绝该帧模型 |
| `reject_angle_deg` | `18.0` | 相对上一帧稳定桌面法向角度跳变超过该值时拒绝该帧模型 |
| `reject_size_ratio` | `1.8` | 自动估计桌面长宽相对上一帧突变超过该比例时拒绝该帧模型 |
| `hold_on_failure` | `true` | 单帧拟合失败时是否短暂沿用上一帧稳定桌面模型 |
| `max_hold_frames` | `8` | 连续拟合失败时最多沿用上一帧模型的帧数 |

`fallback`：平面拟合失败后的旧估计逻辑。

| 键 | 示例 | 说明 |
| --- | --- | --- |
| `dynamic_scale` | `0.8` | 平面拟合失败回退旧估计时的桌面范围缩放系数 |

***

### ocr 文字识别

[config/ocr/paddle.yaml](config/ocr/paddle.yaml)。对应 [PaddleOcr](core/components/ocr/paddle_ocr.py)：底层是一份自迁移过来、自包含的 PaddleOCR v5 ONNX 引擎（[core/components/ocr/paddleocr/](core/components/ocr/paddleocr/)，纯 onnxruntime 推理，含 DB 文本检测 + 方向分类 + CTC 识别）。业务流程为：对上游检出的候选框（默认 `Book`）逐个裁剪 → OCR 识别文本并拼接 → 用 `rapidfuzz` 与 `class_registry.ocr_templates` 的模板串模糊匹配 → 命中则产出对应书本物品名称的检测。

候选类别、输出类别与模板文本都归属共享的 `class_registry`（`ocr_candidate_classes` / `ocr_output_classes` / `ocr_templates`），因此本文件只配置 OCR 实现参数与引擎参数。

| 键                 | 示例                    | 说明                                                             |
| ----------------- | --------------------- | -------------------------------------------------------------- |
| `type`            | `paddle`              | OCR 类型                                                         |
| `use_angle_cls`   | `true`                | 是否启用文字方向(0/180)分类矫正                                            |
| `use_gpu`         | `false`               | 是否用 GPU 推理（`false` 走 CPU onnxruntime）                          |
| `enlarge`         | `1.0`                 | 裁剪图放大倍数；`>1` 时先放大再 OCR，有助于识别小字                                 |
| `min_match_score` | `0.0`                 | rapidfuzz 相似度阈值（0~100），超过才认定命中                                 |
| `engine`          | 见下                    | 传给底层 PaddleOCR 引擎的参数                                           |

`engine` 子项（相对路径按运行时工作目录解析为绝对路径）：

| 键                    | 示例                              | 说明                    |
| -------------------- | ------------------------------- | --------------------- |
| `backend`            | `auto`                          | 推理后端：`auto` / `onnx` / `acl`。`auto` 时香橙派用 `acl`（`.om`），其它平台用 `onnx`（`.onnx`）。可被 `3DCV_OCR_BACKEND` 覆盖 |
| `det_model_dir`      | `models/ppocrv5/det.onnx`       | DB 文本检测模型；路径可写 `.onnx`，ACL 后端会自动替换为同名 `.om` |
| `rec_model_dir`      | `models/ppocrv5/rec.onnx`       | SVTR_LCNet 文本识别模型；ACL 后端会自动替换为同名 `.om` |
| `cls_model_dir`      | `models/ppocrv5/cls.onnx`       | 文本方向(0/180)分类模型；ACL 后端会自动替换为同名 `.om` |
| `rec_char_dict_path` | `models/ppocrv5/ppocrv5_dict.txt` | 识别字符字典                |
| `det_db_box_thresh`  | `0.3`                           | DB 检测框置信度阈值           |
| `det_box_type`       | `quad`                          | 检测框类型：`quad`（四点）/ `poly`（多边形） |

ACL `.om` 后端会从模型输入 shape 自动推导预处理尺寸和 batch。例如当前远端 OCR `.om` 为：`det [1,3,960,960]`、`rec [1,3,48,320]`、`cls [1,3,80,160]`，代码会自动把 det resize 到固定尺寸，并把 rec/cls batch 限制为 1。

> `class_registry.ocr_templates` 每一项是该类别的关键词集合串（例如「数学书」对应「高等数学线性代数函数…」）；OCR 把书上所有文字拼成一串后，与这些模板做模糊匹配，取相似度最高者。要新增/调整书本类别，需同时改 `class_registry` 的 `result_classes` / `result_class_to_goal_id` / `ocr_output_classes` / `ocr_templates`。

***

### counter 计数器

[config/counter/bayesian.yaml](config/counter/bayesian.yaml)。对应 [BayesianCounter](core/components/counter/bayesian_counter.py)：把「每帧观测数量」视为真实数量在漏检/误检噪声下的带噪观测，逐帧用贝叶斯后验估计每类最可能的数量，并在超过总数上限时按置信度削减。

| 键                     | 示例         | 说明                                             |
| --------------------- | ---------- | ---------------------------------------------- |
| `type`                | `bayesian` | 计数器类型                                          |
| `max_object_count`    | `5`        | 单类别数量上限（后验分布在 `0..max_object_count` 上估计）       |
| `total_max`           | `15`       | 所有类别数量之和的上限；超出时优先削减「减 1 后置信度损失最小」的类别           |
| `miss_rate`           | `0.7`      | 漏检率（似然模型参数）                                    |
| `false_positive_rate` | `0.1`      | 误检率（似然模型参数）                                    |
| `smooth_window`       | `5`        | 滑动窗口大小，用于取稳健观测值                                |
| `min_positive_frames` | `3`        | 窗口内正样本帧数下限；不足则该类别记 0（会被限制到不超过 `smooth_window`） |
| `selection_threshold` | `0.5`      | 后验最大值需超过该置信度才计为对应数量，否则记 0                      |

***

## 环境变量

| 变量                               | 作用                                                       |
| -------------------------------- | -------------------------------------------------------- |
| `3DCV_PLATFORM`                  | 覆盖平台探测结果（如 `macos` / `orangepi`），影响后端选择与 `base_paths` 解析 |
| `3DCV_YOLO_BACKEND`              | 覆盖 YOLO 推理后端（`auto` / `onnx` / `acl`）                    |
| `3DCV_OCR_BACKEND`               | 覆盖 PaddleOCR 推理后端（`auto` / `onnx` / `acl`）               |
| `3DCV_RESULT_DIR`                | 覆盖裁判结果文件的落盘目录                                            |
| `OPENNI2_REDIST` / `OPENNI2_LIB` | OpenNI2 库路径（`openni_lib=null` 时用于自动探测）                   |

***

## 日志与产物

- **日志**：写入 `logging.base_dir`（默认 `runs/logs/`），文件名带时间戳与轮次名。事件为结构化文本，形如 `state_enter | state=PROCESS_TABLE_1_WINDOW | table=1`。开启 `logging.per_frame` 会额外记录每帧的流水线细节。
- **结果文件**：写入 `referee.result_base_dir`（或 `3DCV_RESULT_DIR`），文件名 `{file_prefix}-R{轮次}.txt`。识别过程中会持续写入「最新结果」，比赛正常结束或被中断时都会保存最终结果。

***

> `scripts/` 目录下是相机标定、深度到彩色对齐（D2C）、评测视频对齐等**独立调试脚本**，不属于主运行链路，按需单独使用。
