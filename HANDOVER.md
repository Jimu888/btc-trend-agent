# 项目交接文档 · BTC Trading Agent

> **给接手的 AI/Claude 看的**：这是 Jimu888 正在做的两个互锁的 BTC 交易项目。读完本文件你应该能立刻接手继续。
> **生成时间**：2026-04-22

---

## 🎯 项目全景（30 秒）

两个互相配合的系统：

```
┌────────────────────────────────┐      ┌─────────────────────────────┐
│   BTC Trend Agent (v5f)        │      │   BTC Cycle Monitor         │
│   每日决策：BUY/SELL/HOLD       │ ←──  │   每周决策：仓位大小         │
│   目标：吃 BTC 大趋势           │      │   目标：判断大周期阶段        │
│   状态：回测完成，准备实盘       │      │   状态：5 轴能跑，部分校准     │
└────────────────────────────────┘      └─────────────────────────────┘
         ↑ mechanical                          ↑ macro overlay
```

两者都基于我们四轮分析一个传奇 BitMEX 交易员（Paul Wei / `@coolish`）的 6 年 17 万笔成交得出的结论。

---

## 🏆 核心成果摘要

### Agent v5f（决策逻辑）

**只有两条规则**：
- **入场**：`close > MA150 AND close > MA50 AND MA50(今天) > MA50(10 天前)`
- **出场**：`close < MA150`
- 满仓、无止损、无止盈、不做空、不加仓

**历史回测**（2020-05 → 2026-04，含 0.1% 手续费 + 0.05% 滑点）：
- 收益 **+821%**（BTC 买持 +324%）
- 最大回撤 **-44%**（BTC 买持 -77%）
- 滚动 2 年窗口 **7/7** 盈利
- 样本外优于样本内

### Cycle Monitor（5 轴框架）

每周扫描 5 个维度，合成 -10 到 +10 分数：

| 轴 | 数据源 | 覆盖 |
|---|---|---|
| 1. 周期时钟 | 本地日期计算 | ✅ 完整 |
| 2. 链上基本面 | mempool.space, blockchain.info | ⚠️ 部分（MVRV/SOPR 需 Glassnode 付费）|
| 3. 衍生品 | Binance 公开 API | ✅ 完整 |
| 4. 宏观 | FRED + yfinance | ✅ 完整 |
| 5. 情绪 | alternative.me + CoinGecko | ✅ 完整 |

**最新读数（2026-04-21）**：**-2（模糊区）**
- 减半后 24 月，价格从 $106k 回调 28% 到 $76k
- 多个逆向看涨信号（F&G 33、funding 负、基差 -0.06%）
- 但未达深熊特征，建议让 v5f 决定，仓位 50% 无杠杆

---

## 🗂️ 文件位置

### 本地

```
/Users/jimu/btc-trader-analysis/        ← Paul Wei 原始分析项目
├── scripts/ (27 个分析脚本)
├── data/ (raw + klines + derived CSVs, ~200MB)
└── deliverables/ (agent_v5f.py, BRIEFING.md, runbook.py)

/Users/jimu/btc-trend-agent-repo/        ← GitHub 仓库的本地镜像
└── 镜像 https://github.com/Jimu888/btc-trend-agent

/Users/jimu/btc-cycle-agent/             ← 周期监控
├── weekly_cycle_check.py                ← 主脚本
├── .env                                  ← FRED key (gitignored)
├── cache/cycle_*.json                    ← 每次运行的快照
├── HANDOVER.md                           ← 本文件
└── .agents/skills/                       ← Binance skills hub 14 个 skills
```

### GitHub
- **https://github.com/Jimu888/btc-trend-agent** — 公开，已含 v5f 核心 + 完整分析 + 27 脚本
- 周期监控代码计划加入此 repo 的 `cycle-monitor/` 子目录

---

## 🚦 关键决策与状态（时间顺序）

| 时点 | 决策/事件 | 现状 |
|---|---|---|
| 分析开始 | 用户想解码 Paul Wei 的交易方法 | — |
| 四轮分析 | 尝试 v2/v3/v4/v5 各种复杂策略 | 复杂版全部大亏，简单版胜出 |
| v5f 敲定 | MA150+slope10 入场，MA150 出场 | 回测 +821% |
| 参数验证 | 20 组参数敏感性 + 滚动窗口 | 中位 +384%，全部盈利 |
| 时间级测试 | 1h/4h/日/周 | 按日历时长缩放通用，5m 不行 |
| 5x 杠杆决定 | 用户明知风险坚持 5x | 已加 3 条额外护栏（-12% 强平等） |
| GitHub 发布 | 公开 Jimu888/btc-trend-agent | ✅ |
| 转向周期分析 | 用户问"大周期怎么判断" | 5 轴框架 + weekly_cycle_check.py 完成 |
| FRED 配置 | 用户贴了 key（已存 .env） | ⚠️ 用户泄露过一次，建议 regenerate |
| 实盘部署 | 用户计划喂给 Binance AI Pro | **未完成**，待接手 |

---

## 📋 未完成事项（优先级排序）

### P0 — 直接影响实盘
1. **正式实盘部署**（等用户在 Binance AI Pro 操作）
   - API key 已创建（最小权限，IP 白名单）
   - 5x 杠杆、$900 USDT
   - 需要把 `/Users/jimu/btc-trader-analysis/deliverables/runbook.py` 接入 Binance 合约 API
   - 先跑 2 周 DRY_RUN
2. **Regenerate FRED API key**
   - 旧的被贴进对话里（虽然只读公开数据，但习惯问题）
   - 新 key 更新到 `/Users/jimu/btc-cycle-agent/.env`

### P1 — 提升监控质量
3. **修 cycle monitor 的 3 个校准 bug**：
   - yfinance DX-Y.NYB timeout fallback 误扣分
   - 算力 30pt 变化扣分阈值过敏感
   - 周期时钟 21-30 月一刀切"熊市"，应考虑价格是否仍在 MA200W 之上
4. **注册 Glassnode 免费账号拿 API key**，把轴 2 链上覆盖从 40% 提到 80%
5. **把 cycle monitor 加入 GitHub repo**（作为 `cycle-monitor/` 子目录）

### P2 — 长期迭代
6. **每周 cycle check 的定时任务**（launchd/cron，每周一 UTC 01:00 自动跑）
7. **cycle score 反馈到 v5f 仓位决策**的联动逻辑
8. **Gemma Vision 的用法再实验**（200 批否决了"AGREE = 质量"假设，但 SHORT 少样本异常高胜率可能有真信号，需要 500+ 样本确认）

---

## ⚠️ 硬性约束（禁止跨越）

这些来自分析结论，接手者**必须遵守**：

1. **不改 v5f 策略**（MA150/slope10 经过验证，任何改动都需先跑历史回测）
2. **不加止损/止盈/pyramid/martingale** —— 回测证明全部让收益下降
3. **不在 1m/5m 级别跑 v5f** —— 已证明亏钱（手续费吃干 edge）
4. **不复制参数到其他币种** —— 未验证
5. **不私自提高杠杆**（5x 已是用户明知风险的决定，不要因为看起来"能赚更多"而提到 10x）
6. **不把 Paul 的做 T 风格加回来** —— 他的做 T 是净负贡献（martingale 亏 17 BTC，给回利润亏 37 BTC）

---

## 💡 和用户打交道的要点

- **用户是非技术 PM，中文沟通**
- **偏好诚实承认局限**（见 `~/.claude/projects/-Users-jimu/memory/feedback_prefer_honest_limits.md`）
- **遇到质疑先坦诚**，不要防御性辩护
- **实证优先**：如果用户想改什么，跑数据说话
- **他会接受"归零"风险**但前提是你已经充分提示
- **不要粘 API key 到对话里**——他已经这么做过一次

---

## 🔗 关键外部链接

- **Paul 原始数据**：https://github.com/bwjoke/BTC-Trading-Since-2020
- **v5f 策略 repo**：https://github.com/Jimu888/btc-trend-agent
- **FRED 数据**：https://fred.stlouisfed.org（需 API key）
- **Binance 公开 API**：https://fapi.binance.com/fapi/v1/（不需要 auth）
- **Fear & Greed**：https://alternative.me/crypto/fear-and-greed-index/
- **Glassnode**：https://glassnode.com（付费 $39+/月或免费 tier）
- **Binance Skills Hub**：https://github.com/binance/binance-skills-hub

---

## 🎬 接手的第一步

1. **读这份 HANDOVER.md**（✅ 你正在读）
2. **读 GitHub repo 的 `analysis/ANALYSIS_REPORT.md`**，理解为什么 v5f 是这个样子
3. **读 `BRIEFING.md`**，理解行为约束
4. **问用户当前目标**：
   - 推进实盘？（P0 第 1 条）
   - 改进 cycle monitor？（P1）
   - 别的方向？
5. **不要立刻动代码**，先确认理解

祝你顺利接手。
