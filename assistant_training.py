import json
import random
from datetime import datetime, timedelta

statuses = [
    "lower than usual",
    "normal",
    "above normal",
    "extremely high"
]

trends = ["rising", "falling", "stable"]

base = datetime(2026, 7, 10, 12, 0)

examples = []

for i in range(50):

    current = round(random.uniform(0.4, 7.5), 2)

    trend = random.choice(trends)

    if trend == "rising":
        change = round(random.uniform(0.01, 0.60), 2)
    elif trend == "falling":
        change = round(-random.uniform(0.01, 0.60), 2)
    else:
        change = 0.00

    maximum = round(current + max(change, 0) + random.uniform(0, 0.20), 2)
    minimum = round(current + min(change, 0) - random.uniform(0, 0.20), 2)

    percentile = round(random.uniform(0, 100), 1)
    status = random.choice(statuses)

    last_updated = base + timedelta(hours=i)
    peak_time = last_updated + timedelta(hours=random.randint(1, 48))

    system = {
        "currentLevel": current,
        "meanPercentile": percentile,
        "meanStatus": status,
        "trend": trend,
        "change": change,
        "lastUpdated": last_updated.isoformat(),
        "maximumForecast": maximum,
        "minForecast": minimum,
        "peakTime": peak_time.isoformat()
    }

    if trend == "rising":
        summary = (
            f"The current water level is at {current:.2f} m. "
            f"The water level is projected to be rising over the course of the next 48 hours by {change:.2f} m. "
            f"This indicates that the water level is {status} when compared to previous samples. "
            f"Over the next 48 hours, it will reach its highest point of {maximum:.2f} m at "
            f"{peak_time.strftime('%B %d, %Y at %I:%M %p')}."
        )

    elif trend == "falling":
        summary = (
            f"The current water level is at {current:.2f} m. "
            f"The water level is projected to be falling over the course of the next 48 hours by {change:.2f} m. "
            f"This indicates that the water level is {status} when compared to previous samples. "
            f"Over the next 48 hours, it will reach its highest point of {maximum:.2f} m at "
            f"{peak_time.strftime('%B %d, %Y at %I:%M %p')}."
        )

    else:  # stable
        summary = (
            f"The current water level is at {current:.2f} m. "
            f"The water level is projected to remain stable over the course of the next 48 hours with little overall change. "
            f"This indicates that the water level is {status} when compared to previous samples. "
            f"Over the next 48 hours, it will reach its highest point of {maximum:.2f} m at "
            f"{peak_time.strftime('%B %d, %Y at %I:%M %p')}."
        )
        
    examples.append([
        {
            "role": "system",
            "content": json.dumps(system)
        },
        {
            "role": "user",
            "content": "Write a concise summary of the data."
        },
        {
            "role": "assistant",
            "content": summary
        }
    ])

print(json.dumps(examples, indent=2))