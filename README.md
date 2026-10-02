# GridPilot-AGH

面向含光伏与储能配电网的滚动优化调度、安全校核和异常修复智能体原型。

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
- 基于HiGHS稀疏确定性等价模型的真正两阶段场景优化：一阶段共享日前购电、储能基准计划及备用容量，二阶段为每个场景分别建立电网平衡、储能SOC、需求响应、弃光和失负荷追索变量；
- 显式非预见性、95% CVaR线性化、最坏场景鲁棒目标，以及确定性日前计划、风险厌恶随机策略和鲁棒策略的同场景比较；
- 购电成本、储能损耗、弃光和失负荷综合目标；
- pandapower基态AC潮流与线路故障后的孤岛、低电压和过载识别；
- AC校核越限后的电压裕度/线路安全系数反馈与重新优化；
- 正常、光伏预测偏差、储能退出、主干线路降额，以及“投资建设后”五类场景；
- 投资建设后场景显式投运2条新建独立备用馈线、18段关键走廊与5条联络线增容降阻、3×600 kvar SVG和主变OLTC，并重新执行基态AC与768个N-1工况；
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

浏览器打开 `http://127.0.0.1:8765`，选择场景后可先检查“预测不确定性输入”。开启“Agent 接管参数”后，系统会根据当前场景自动替换误差分布和风险偏好；关闭后可手工修改。点击“开始演示”后，网页会显示模拟处理进度，并读取 `output/<场景>/results.json` 中的预计算调度结果，同时在线生成不确定性场景和风险比较。“投资建设后”还会展示投资清单、建设前后指标对比和标出新建设备的拓扑图。

结果页包含概率扇形图、20条场景轨迹、误差直方图、相关系数矩阵、三类优化策略比较、迭代SCOPF过程、事故后纠正资源动作、风险热力图、联络/备用馈线恢复拓扑、修复时间轴、功率/SOC/电压/线路曲线、调度计划和执行轨迹。

不确定性模块使用大规模稀疏LP的确定性等价形式。所有场景共享同一组一阶段变量，场景揭示后才允许使用独立追索变量；风险厌恶目标采用CVaR辅助变量和尾部超额变量线性化。模型结构参考 [PyPSA stochastic optimization](https://docs.pypsa.org/latest/user-guide/optimization/stochastic/) 的两阶段与CVaR表述，项目实现仍直接使用SciPy HiGHS，不依赖PyPSA运行时。

Agent 与人工输入使用同一接口：`POST /api/uncertainty`；可替换字段包括光伏、负荷、价格误差，场景数，CVaR置信度，风险厌恶，误差分布，时序相关系数和随机种子。

## 三类比赛验证样例

```powershell
.\.venv\Scripts\python.exe run_demo.py --scenario normal --output output/normal
.\.venv\Scripts\python.exe run_demo.py --scenario pv_error --output output/pv_error
.\.venv\Scripts\python.exe run_demo.py --scenario battery_outage --output output/battery_outage
.\.venv\Scripts\python.exe run_demo.py --scenario line_derating --output output/line_derating
.\.venv\Scripts\python.exe run_demo.py --scenario post_investment --output output/post_investment
```

- 正常：基准预测和设备状态；
- 边界：光伏出力仅为预测值的 65%；
- 失败：光伏偏低、负荷增加且储能退出，系统必须明确报告降级或越限。
- 规划：投资资产投运后保持相同安全阈值，完整N-1安全率由现状网架的64.84%提升到99.09%（761/768安全）。

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

这是赛事原型，不直接用于真实电网控制。当前pandapower校核采用三相平衡稳态模型；线路N-1高风险约束已经通过三轮约束生成反馈到滚动调度，但每个故障仍只选择一条联络线或一条备用馈线，尚未实现多开关网络重构，也未覆盖保护动作和暂态稳定。默认3小时物理抢修是可配置演示参数，并非城市配电故障的统一承诺时限。两阶段随机模型已显式包含共享一阶段和逐场景追索变量，但场景内网络约束采用基于AC校核结果标定的电压/热稳定安全包络，而不是为每个场景复制完整33节点AC潮流方程。真实部署前应接入经校核的 EMS/DMS 数据、调度规则及人工审批机制。
