# Demo Video Script
Supreme System V5 - 5-Minute Product Demo

---

## 🎬 Script Overview

**Duration**: 5 minutes
**Target Audience**: Trading firms, fintech startups, professional traders
**Goal**: Show ease of deployment + key features + clear CTA

---

## 📝 Script (Time-coded)

### [0:00 - 0:30] Introduction

**[Screen: Title slide or desktop]**

> "Hi, I'm [Name] from Supreme System V5.
>
> If you're looking for a production-ready algorithmic trading platform that deploys in minutes - not months - you're in the right place.
>
> Today I'll show you how to go from zero to live paper trading in under 15 minutes.
>
> Let's dive in."

**[Transition to terminal]**

---

### [0:30 - 1:30] Installation & Setup

**[Screen: Terminal with commands]**

> "First, clone the repository and install dependencies.
> 
> This is literally three commands:"

```bash
git clone https://github.com/thanhmuefatty07/supreme-system-v5.git
cd supreme-system-v5
pip install -r requirements.txt
```

**[Show commands running - fast forward if needed]**

> "Copy the environment template and configure your settings."

```bash
cp .env.example .env
# Edit .env - show TRADING_MODE=paper
```

> "That's it. Setup complete in 2 minutes."

---

### [1:30 - 2:30] Dashboard Demo

**[Screen: Streamlit dashboard]**

> "Now let's start the interactive dashboard."

```bash
streamlit run src/monitoring/dashboard.py
```

**[Show dashboard loading, then main interface]**

> "Here's the dashboard. You can see:
> 
> - Real-time portfolio value and positions
> - Active strategies and signals
> - Risk metrics and circuit breaker status
> - Recent trades and P&L
>
> Everything updates in real-time."

**[Click through key dashboard sections]**

---

### [2:30 - 3:30] Paper Trading Demo

**[Screen: Terminal]**

> "Let's run a paper trading simulation with real market data."

```bash
python -m src.cli paper-trade \
    --symbols AAPL MSFT GOOGL \
    --capital 100000 \
    --strategy momentum
```

**[Show command running, output streaming]**

> "The system is now:
> 
> - Fetching real market data
> - Calculating technical indicators
> - Generating trading signals
> - Executing simulated trades
> - Monitoring risk in real-time
>
> All without touching real money."

**[Show 10-15 seconds of output, highlight key events]**

---

### [3:30 - 4:30] Monitoring & Metrics

**[Screen: Split - Prometheus + Grafana]**

> "For production deployments, we include full monitoring stack."

**[Show Prometheus metrics page]**

> "Prometheus collects all system metrics:
> - Trade execution times
> - Risk check results  
> - Portfolio values
> - System health"

**[Switch to Grafana]**

> "Grafana dashboards give you real-time visibility:
> 
> - Performance charts
> - Risk alerts
> - Strategy performance
> - System resource usage
>
> All production-ready, no configuration needed."

---

### [4:30 - 5:00] Call to Action

**[Screen: Contact slide or back to desktop]**

> "That's Supreme System V5.
>
> To recap:
> - Deploys in 15 minutes
> - Production-grade risk management
> - Full monitoring and observability
> - Starting at $10,000 for commercial license
>
> **Ready to try it yourself?**
>
> We offer a free 30-day evaluation license.
> No credit card required.
>
> Email thanhmuefatty07@gmail.com
> Subject: 'Evaluation License Request'
>
> Or visit [your website/GitHub]
>
> Thanks for watching, and happy trading!"

**[End screen: Logo + contact info for 3-5 seconds]**

---

## 🎥 Production Tips

### Before Recording

- [ ] Test all commands in clean environment
- [ ] Prepare sample data for quick demo
- [ ] Clear terminal history and unnecessary windows
- [ ] Use large, readable terminal font (18-24pt)
- [ ] Enable Do Not Disturb (no notifications)
- [ ] Have backup plan if live demo fails

### Recording Setup

**Software**: OBS Studio (free) or ScreenFlow (Mac)
**Resolution**: 1920x1080 (Full HD)
**Frame Rate**: 30 FPS minimum
**Audio**: Clear microphone (Blue Yeti, Rode, or lapel mic)
**Editing**: DaVinci Resolve (free) or Adobe Premiere

### After Recording

- [ ] Add captions/subtitles (YouTube auto-generate)
- [ ] Add chapter markers at each section
- [ ] Create thumbnail with key benefit
- [ ] Upload to YouTube (unlisted) and Vimeo
- [ ] Embed on website/landing page
- [ ] Share link in outreach emails

---

## 🎯 Variations

**3-Minute Version**: Skip detailed dashboard walkthrough
**10-Minute Version**: Add backtest demo + more technical details
**20-Minute Version**: Add strategy customization + API integration

---
**Last Updated**: November 16, 2025
