# Calendly Integration Setup Guide

## Overview

This guide helps you set up Calendly integration for Supreme System V5 demo scheduling and sales inquiries.

---

## Step 1: Create Calendly Account

1. Go to [calendly.com](https://calendly.com)
2. Sign up for free account
3. Complete profile setup

---

## Step 2: Create Event Types

### Demo Call Event

**Settings:**
- **Name**: "Supreme System V5 Demo"
- **Duration**: 30 minutes
- **Buffer**: 15 minutes between meetings
- **Location**: Video call (Zoom/Google Meet)

**Questions to Ask:**

1. Company name
2. Trading volume (monthly)
3. Current tech stack
4. Primary use case
5. Budget range

### Sales Consultation Event

**Settings:**
- **Name**: "Supreme System V5 Sales Consultation"
- **Duration**: 45 minutes
- **Buffer**: 15 minutes between meetings
- **Location**: Video call

---

## Step 3: Integration Code

### Landing Page Integration

Add to `index.html`:

```html
<!-- Calendly inline widget -->
<div class="calendly-inline-widget" 
     data-url="https://calendly.com/your-username/supreme-system-v5-demo" 
     style="min-width:320px;height:630px;"></div>
<script type="text/javascript" src="https://assets.calendly.com/assets/external/widget.js" async></script>
```

### Button Integration

Replace demo button in `index.html`:

```html
<a href="https://calendly.com/your-username/supreme-system-v5-demo" 
   class="cta-button" 
   target="_blank">
   📞 Schedule Demo
</a>
```

---

## Step 4: Email Templates

### Confirmation Email Template

**Subject:** Thanks for booking a Supreme System V5 demo!

**Body:**

```
Hi {{invitee_name}},

Thanks for booking a demo of Supreme System V5!

Before our call, here's what to expect:

- Live system walkthrough (10 min)
- Architecture deep-dive (10 min)
- Q&A (10 min)

To prepare:

- Try the demo: https://github.com/thanhmuefatty07/supreme-system-v5
- Review docs: https://your-docs-url
- Prepare questions

See you soon!

[Your Name]
Supreme System V5 Team
```

### Reminder Email Template

**Subject:** Reminder: Supreme System V5 Demo Tomorrow

**Body:**

```
Hi {{invitee_name}},

Just a reminder about our demo call tomorrow at {{event_start_time}}.

Looking forward to showing you Supreme System V5!

[Your Name]
```

---

## Step 5: Update Links

Update these files với Calendly link:

1. **README.md**: Demo request link
2. **index.html**: CTA buttons
3. **docs/commercial/licensing.md**: Demo scheduling link
4. **DEMO.md**: Demo call link

---

## Step 6: Analytics Setup

1. Enable Calendly analytics
2. Track conversion rates
3. Monitor no-shows
4. Analyze booking patterns

---

## Best Practices

- **Response Time**: Respond to bookings within 2 hours
- **Follow-up**: Send follow-up email after demo
- **Availability**: Set realistic availability windows
- **Buffer Time**: Always include buffer between meetings

---

## Troubleshooting

### Widget not loading?

- Check Calendly URL is correct
- Verify JavaScript is enabled
- Check browser console for errors

### Bookings not showing?

- Verify Calendly account is active
- Check email notifications are enabled
- Verify calendar sync is working

---

**Need help?** Contact: thanhmuefatty07@gmail.com

