# Ananya-Dutta
[README (1).md](https://github.com/user-attachments/files/27307697/README.1.md)
#  MindfulAI — Companion for Teen Mental Wellness

> *An AI-powered mental health companion built by a teenager, for teenagers.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square)]()
[![Made with by Ananya](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20by-Ananya%20Dutta-ff69b4?style=flat-square)]()

---

##  Why This Exists

1 in 5 teenagers experiences a diagnosable mental health condition. Yet most mental health apps are designed by adults, for adults — clinically cold, expensive, and inaccessible to the people who need them most.

I'm Ananya, a high school student from West Bengal, India. I built MindfulAI because I wanted a tool that *actually* speaks to teenagers: something private, judgment-free, and grounded in real emotional science — not just generic wellness advice.

MindfulAI is not a replacement for therapy. It is a **first step** — a safe space to process feelings, track moods, and receive empathetic, evidence-informed support at 2am when there's no one else to talk to.

---

##  Features

| Feature | Description |
|---|---|
|  **AI Chat Companion** | Empathetic conversation powered by rule-based NLP + sentiment analysis |
|  **Mood Tracker** | Daily mood logging with trend visualization over time |
|  **Private Journal** | Encrypted local journaling with optional reflection prompts |
|  **Breathing Exercises** | Guided 4-7-8 and box breathing with animated UI |
|  **Wellness Dashboard** | Visual summary of mood patterns, streaks, and insights |
|  **Crisis Resources** | Instant access to helplines (iCall India, Vandrevala, iNSTEP) |
|  **Privacy-First** | All data stored locally — nothing sent to a server |

---

## Architecture

```
mindfulai/
├── app.py                  # Streamlit entry point
├── mindfulai/
│   ├── chat.py             # Conversation engine & response logic
│   ├── sentiment.py        # Sentiment analysis (VADER + keyword rules)
│   ├── mood_tracker.py     # Mood logging & trend analysis
│   ├── journal.py          # Journal entry management
│   ├── breathing.py        # Breathing exercise logic
│   ├── crisis.py           # Crisis detection & resource surfacing
│   └── utils.py            # Shared helpers
├── data/
│   ├── responses.json      # Empathy response templates by emotion
│   ├── crisis_keywords.txt # Crisis signal words (for detection)
│   └── prompts.json        # Reflection prompt bank
├── docs/
│   ├── research_article.pdf
│   └── project_report.pdf
├── tests/
│   ├── test_sentiment.py
│   └── test_chat.py
├── requirements.txt
└── README.md
```

---

##  Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/duttaananya552-alt/mindfulai.git
cd mindfulai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (first run only)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

##  How It Works

### Sentiment Analysis Pipeline

MindfulAI uses a hybrid sentiment approach:

1. **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — a lexicon-based model optimized for social media and informal text, well-suited for how teenagers actually write.
2. **Custom keyword rules** — a curated layer that catches teenage slang, colloquialisms, and mental health signals that VADER misses (e.g., "lowkey not okay", "it's whatever", "I don't care anymore").
3. **Crisis detection** — a separate keyword pass scans every message for high-risk phrases and immediately surfaces appropriate resources if triggered.

```
User message
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│ VADER Score │────▶│ Keyword Override  │────▶│ Crisis Check   │
│  (-1 to +1) │     │ (slang + rules)   │     │ (safety first) │
└─────────────┘     └──────────────────┘     └────────────────┘
     │                       │                       │
     └───────────────────────┴───────────────────────┘
                             │
                      Emotion Category
                   (joy / sadness / anxiety /
                    anger / neutral / crisis)
                             │
                      Response Engine
                   (template + variation +
                    follow-up question)
```

### Mood Tracking

Moods are stored as a local JSON file with timestamps. The dashboard uses **Plotly** to render:
- A 7-day mood timeline
- Weekly average trend line
- Most frequent emotional state

---

##  Research Foundation

MindfulAI's response design draws on:
- **Cognitive Behavioral Therapy (CBT)** — identifying thought patterns and offering gentle reframes
- **Motivational Interviewing** — open-ended questions that encourage self-reflection rather than giving advice
- **Positive Psychology** — strength-based language, gratitude prompts, small wins

> 📄 Read the full research article: [`docs/research_article.pdf`]https://drive.google.com/file/d/1xwomrskHZgU5SJUZHI66cFOtMwi38XfI/view?usp=sharing
> 📘 Read the project report:https://drive.google.com/file/d/1eZwaadPP9ZJaYmA4QMYuTEFFRLpwBh4l/view?usp=sharing



---
"""
test_chat.py — Tests for the chat engine.
Run with: python -m pytest tests/
"""

import pytest
from mindfulai.chat import ChatEngine


@pytest.fixture
def engine():
    return ChatEngine()


class TestResponseGeneration:
    def test_returns_string(self, engine):
        response, result = engine.respond("I'm feeling okay today")
        assert isinstance(response, str)
        assert len(response) > 10

    def test_crisis_response_mentions_helpline(self, engine):
        response, result = engine.respond("I want to kill myself")
        assert result.is_crisis is True
        assert "9152987821" in response or "iCall" in response

    def test_turn_counter_increments(self, engine):
        assert engine.turn_count == 0
        engine.respond("Hello")
        assert engine.turn_count == 1
        engine.respond("I'm sad")
        assert engine.turn_count == 2

    def test_reset_clears_state(self, engine):
        engine.respond("Hello")
        engine.respond("I'm anxious")
        engine.reset()
        assert engine.turn_count == 0
        assert engine.dominant_emotion is None

    def test_dominant_emotion(self, engine):
        engine.respond("I'm so sad")
        engine.respond("I feel really low")
        engine.respond("Everything is hard")
        # Dominant should be a known emotion
        assert engine.dominant_emotion in {
            "sadness", "anxiety", "anger", "joy",
            "neutral", "overwhelmed", "lonely", "crisis"
        }

##  Privacy & Safety"""
mood_tracker.py — Mood logging and trend analysis for MindfulAI.

Stores mood entries in a local SQLite database. No data leaves the device.
"""
[requirements.txt](https://github.com/user-attachments/files/27307718/requirements.txt)
import sqlite3[test_chat.py](https://github.com/user-attachments/files/27307722/test_chat.py)
import datetime
from dataclasses import dataclass[test_chat.py](https://github.com/user-attachments/files/27307709/test_chat.py)
[crisis.py](https://github.com/user-attachments/files/27307715/crisis.py)
from pathlib import Path
from typing import Optional[test_sentiment.py](https://github.com/user-attachments/files/27307710/test_sentiment.py)


DB_PATH = Path.home() / ".mindfulai" / "moods.db"

MOOD_LABELS = {
    1: "Very low",
    2: "Low",
    3: "Okay",
    4: "Good",
    5: "Great",
}

MOOD_COLORS = {
    1: "#E57373",
    2: "#FFB74D",
    3: "#FFF176",
    4: "#81C784",
    5: "#4DB6AC",
}


@dataclass
class MoodEntry:
    mood: int              # 1–5
    note: str
    timestamp: datetime.datetime
    emotion_tag: Optional[str] = None


class MoodTracker:
    """
    Records daily moods and provides trend analysis.

    Usage:
        tracker = MoodTracker()
        tracker.log(mood=4, note="Had a good study session", emotion_tag="joy")
        entries = tracker.get_last_n_days(7)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def log(
        self,
        mood: int,
        note: str = "",
        emotion_tag: Optional[str] = None,
    ) -> MoodEntry:
        """Log a mood entry. mood must be 1–5."""
        if not 1 <= mood <= 5:
            raise ValueError(f"Mood must be between 1 and 5, got {mood}")

        now = datetime.datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO moods (mood, note, emotion_tag, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (mood, note, emotion_tag, now.isoformat()),
            )

        return MoodEntry(mood=mood, note=note, timestamp=now, emotion_tag=emotion_tag)

    def get_last_n_days(self, n: int = 7) -> list[MoodEntry]:
        """Return mood entries from the last n days, oldest first."""
        since = datetime.datetime.now() - datetime.timedelta(days=n)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT mood, note, emotion_tag, timestamp
                FROM moods
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (since.isoformat(),),
            ).fetchall()

        return [
            MoodEntry(
                mood=r[0],
                note=r[1],
                emotion_tag=r[2],
                timestamp=datetime.datetime.fromisoformat(r[3]),
            )
            for r in rows
        ]

    def weekly_average(self) -> Optional[float]:
        """Average mood over the last 7 days, or None if no data."""
        entries = self.get_last_n_days(7)
        if not entries:
            return None
        return round(sum(e.mood for e in entries) / len(entries), 2)

    def streak(self) -> int:
        """Number of consecutive days with at least one mood log."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT DATE(timestamp) as day
                FROM moods
                GROUP BY day
                ORDER BY day DESC
                """
            ).fetchall()

        if not rows:
            return 0

        streak = 0
        today = datetime.date.today()

        for i, (day_str,) in enumerate(rows):
            day = datetime.date.fromisoformat(day_str)
            expected = today - datetime.timedelta(days=i)
            if day == expected:
                streak += 1
            else:
                break

        return streak

    def most_common_emotion(self) -> Optional[str]:
        """Most frequently logged emotion tag overall."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT emotion_tag
                FROM moods
                WHERE emotion_tag IS NOT NULL
                GROUP BY emotion_tag
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """
            ).fetchone()
        return row[0] if row else None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS moods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mood INTEGER NOT NULL,
                    note TEXT,
                    emotion_tag TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )


- **No account required.** No email, no sign-up.
- **All data stays on your device.** Journal entries and mood logs are stored in a local SQLite database — nothing is transmitted over the internet.
- **Crisis-first design.** Any message containing crisis signals immediately surfaces verified helplines before any AI response.
- **Disclaimer**: MindfulAI is a supportive tool, not a clinical service. If you or someone you know is in crisis, please reach out to a professional.

###  Indian Crisis Resources
| Resource | Contact |
|---|---|
| iCall (TISS) | 9152987821 |
| Vandrevala Foundation | 1860-2662-345 |
| SNEHI | 044-24640050 |
| iCall WhatsApp | +91 9152987821 |

---

##  Roadmap

- [x] Core chat engine with sentiment analysis
- [x] Mood tracker with visualization
- [x] Private journaling module
- [x] Breathing exercise UI
- [x] Crisis detection & resource surfacing
- [ ] CBT-inspired conversation flows (structured thought records)
- [ ] Peer support matching (anonymous, moderated)
- [ ] Mobile app deployment (Kivy / React Native)
- [ ] Clinical pilot study with school counselors
- [ ] Multilingual support (Bengali, Hindi)

---

## Contributing

Contributions, bug reports, and suggestions are welcome — especially from teenagers who want to shape what this tool becomes.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes
4. Open a pull request

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

##  About the Author

**Ananya Dutta** is a high school student from West Bengal, India, passionate about building technology that serves real human needs. She designs AI-powered applications focused on teenage mental health, publishes research-driven articles, and explores the ethical use of artificial intelligence in healthcare.

- [ananyaduttao82010@gmail.com](mailto:ananyaduttao82010@gmail.com)
-  [github.com/duttaananya552-alt](https://github.com/duttaananya552-alt)

---

## [mood_tracker.py](https://github.com/user-attachments/files/27307708/mood_tracker.py)
[app.py](https://github.com/user-attachments/files/27307701/app.py)
 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

*"Technology should meet people where they are — especially when they are struggling."*
