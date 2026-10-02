# GridPilot-AGH

面向含光伏与储能配电网的滚动优化调度、安全校核和异常修复智能体原型。

项目对应赛事方向：**B2 算法发现与组合优化 + B3 科学计算与仿真 + B4 科研 Agent 与 Harness 工程 + D4 环境能源与城市系统**。

后续改进与版本路线见 [docs/PROJECT_IMPROVEMENT_ROADMAP.md](docs/PROJECT_IMPROVEMENT_ROADMAP.md)。

## 已实现能力

- 改进 IEEE 33 节点径向配电网与 24 小时负荷、光伏和电价数据；
- 基于 SciPy HiGHS 的12小时滚动时域优化，每小时执行窗口首个控制量；
- 显式嵌入节点有功/无功平衡、LinDistFlow电压降和线路容量约束；
- 建模储能变流器有功/无功能力边界，并以独立AC潮流反馈校准安全裕度；
- 使用 pandapower 牛顿-拉夫逊 AC 潮流进行24小时基态复核；
- 对24小时逐时枚举32条运行线路，完成768个线路N-1工况，并搜索能够跨接故障孤岛的联络线或独立备用馈线；
- 执行“N-1筛查—约束生成—滚动重优化—AC复核”的三轮迭代SCOPF，保留逐轮风险变化证据；
- 在事故后模型中增加储能有功/无功、需求响应和上级电网备用纠正变量；
- N-1风险热力图、迭代约束证据链、事故后资源动作表及故障前后联络/备用馈线恢复拓扑图；
- 区分分钟级故障隔离与临时转供、小时级物理抢修；默认物理抢修为可配置的3小时城市简单故障演示假设；
- 可由用户或Agent填写的预测不确定性输入，生成20个相关时序场景；
- 确定性、两阶段随机和鲁棒策略的期望成本、95% CVaR、失负荷概率与电压越限概率比较；
- 购电成本、储能损耗、弃光和失负荷综合目标；
- pandapower基态AC潮流与线路故障后的孤岛、低电压和过载识别；
- AC校核越限后的电压裕度/线路安全系数反馈与重新优化；
- 正常、光伏预测偏差、储能退出、主干线路降额四类场景；
- 网页演示工作台、HTML 报告、CSV 调度表和 JSON 运行证据；
- 可被 Agnes Harness 调用的 MCP 工具服务；
- 无需安装 Web 框架或专业商业求解器。

## 快速运行

项目依赖 NumPy 和 SciPy（HiGHS随SciPy提供）。建议使用 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_demo.py --scenario normal
```

结果保存在 `output/latest/`：

- `report.html`：可视化运行报告；
- `schedule.csv`：24 小时调度计划；
- `results.json`：完整指标、潮流结果与执行轨迹。

## 启动网页工作台

```powershell
.\.venv\Scripts\python.exe app.py
```

浏览器打开 `http://127.0.0.1:8765`，选择场景后可先检查“预测不确定性输入”。开启“Agent 接管参数”后，系统会根据正常、光伏偏差、储能退出或线路降额场景自动替换误差分布和风险偏好；关闭后可手工修改。点击“开始演示”后，网页会显示模拟处理进度，并读取 `output/<场景>/results.json` 中的预计算调度结果，同时在线生成不确定性场景和风险比较。

结果页包含概率扇形图、20条场景轨迹、误差直方图、相关系数矩阵、三类优化策略比较、迭代SCOPF过程、事故后纠正资源动作、风险热力图、联络/备用馈线恢复拓扑、修复时间轴、功率/SOC/电压/线路曲线、调度计划和执行轨迹。

Agent 与人工输入使用同一接口：`POST /api/uncertainty`；可替换字段包括光伏、负荷、价格误差，场景数，CVaR置信度，风险厌恶，误差分布，时序相关系数和随机种子。

## 三类比赛验证样例

```powershell
.\.venv\Scripts\python.exe run_demo.py --scenario normal --output output/normal
.\.venv\Scripts\python.exe run_demo.py --scenario pv_error --output output/pv_error
.\.venv\Scripts\python.exe run_demo.py --scenario battery_outage --output output/battery_outage
.\.venv\Scripts\python.exe run_demo.py --scenario line_derating --output output/line_derating
```

- 正常：基准预测和设备状态；
- 边界：光伏出力仅为预测值的 65%；
- 失败：光伏偏低、负荷增加且储能退出，系统必须明确报告降级或越限。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Agnes Harness

接入步骤和推荐提示词见 [docs/AGH_INTEGRATION.md](docs/AGH_INTEGRATION.md)。MCP 服务启动命令：

```powershell
.\.venv\Scripts\python.exe -m gridpilot.mcp_server
```

## 目录

```text
gridpilot-agh/
├─ app.py                    网页工作台入口
├─ run_demo.py               命令行演示入口
├─ data/ieee33.json          电网与设备参数
├─ gridpilot/
│  ├─ data.py                负荷、光伏、电价和场景
│  ├─ optimizer.py           滚动时域LinDistFlow网络约束优化器
│  ├─ powerflow.py           轻量前推回代参考实现（不作为主校核后端）
│  ├─ pandapower_validation.py pandapower AC潮流、孤岛识别与N-1校核
│  ├─ engine.py              Agent 闭环编排
│  ├─ reporting.py           证据和报告生成
│  ├─ webapp.py              零依赖网页端
│  └─ mcp_server.py          AGH/MCP 工具接口
├─ tests/                    自动化测试
└─ output/                   运行产物
```

## 模型边界

这是赛事原型，不直接用于真实电网控制。当前pandapower校核采用三相平衡稳态模型；线路N-1高风险约束已经通过三轮约束生成反馈到滚动调度，但每个故障仍只选择一条联络线或一条备用馈线，尚未实现多开关网络重构，也未覆盖保护动作和暂态稳定。默认3小时物理抢修是可配置演示参数，并非城市配电故障的统一承诺时限。不确定性界面目前采用参数化场景生成与备用策略评估，尚不等同于完整的两阶段随机优化。真实部署前应接入经校核的 EMS/DMS 数据、调度规则及人工审批机制。
