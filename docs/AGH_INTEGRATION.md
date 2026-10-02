# Agnes Harness 接入说明

GridPilot-AGH 提供一个不依赖第三方 MCP SDK 的标准输入输出 MCP 服务。它通过一行一个 JSON-RPC 消息与 Harness 通信。

## 启动命令

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m gridpilot.mcp_server
```

在 Agnes Harness 的 MCP 管理界面新增本地标准输入输出服务，命令填写项目虚拟环境解释器的绝对路径：

```text
C:\Users\Dream\Desktop\0Test\gridpilot-agh\.venv\Scripts\python.exe
```

参数填写：

```text
-m gridpilot.mcp_server
```

工作目录设为本项目根目录。启用并信任服务后，模型可以调用以下工具：

- `validate_grid_case`：检查算例拓扑与参数；
- `run_gridpilot_closed_loop`：执行滚动优化、pandapower AC潮流、线路N-1和修复重优化；
- `compare_grid_scenarios`：比较三类验证场景；
- `export_gridpilot_report`：导出 HTML、CSV 和 JSON 证据。

建议在 AGH 中使用下面的任务提示：

> 请先检查 IEEE 33 节点算例。检查通过后运行正常场景，解释调度结果和安全校核指标；随后运行储能退出场景，对比失负荷、最低电压和异常时段。如果结果需要关注，不得声称校核通过。最后导出正常场景报告。

## 运行证据

比赛证据包至少保留：

1. AGH 的工具调用轨迹截图；
2. `output/latest/results.json`；
3. `output/latest/schedule.csv`；
4. `output/latest/report.html`；
5. 正常、光伏偏差、储能退出三个场景的对比结果。

不要在仓库或证据包中保存 API Key。
