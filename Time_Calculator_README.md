
# Time Calculator

The **Time Calculator** function `add_time` takes a start time, a duration, and an optional starting day of the week, then calculates the end time after the duration has passed. This function can return the day if specified and the number of days that have passed.

## Function Code

```python
def add_time(start, duration, day_of_week=None):
    # Define a dictionary to map days of the week to their positions
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Helper function to convert 12-hour format to 24-hour format
    def time_to_24_hour(time):
        time_parts = time.split()
        am_pm = time_parts[1]
        hour, minute = map(int, time_parts[0].split(':'))

        if am_pm == 'PM' and hour != 12:
            hour += 12
        elif am_pm == 'AM' and hour == 0:
            hour = 0

        return hour, minute

    # Helper function to convert 24-hour format back to 12-hour format
    def time_to_12_hour(hour, minute):
        period = "AM"
        if hour >= 12:
            period = "PM"
        if hour == 0:
            hour = 12
        elif hour > 12:
            hour -= 12
        return f"{hour}:{minute:02d} {period}"

    # Helper function to calculate the new day of the week
    def calculate_new_day(start_day, days_later):
        if start_day:
            start_day_index = days_of_week.index(start_day.capitalize())
            new_day_index = (start_day_index + days_later) % 7
            return days_of_week[new_day_index]
        return None

    # Parse the input times
    start_hour, start_minute = time_to_24_hour(start)
    duration_hour, duration_minute = map(int, duration.split(':'))

    # Add duration minutes and hours
    end_minute = start_minute + duration_minute
    additional_hour = end_minute // 60
    end_minute = end_minute % 60

    end_hour = start_hour + duration_hour + additional_hour
    days_later = end_hour // 24
    end_hour = end_hour % 24

    # Convert back to 12-hour format
    new_time = time_to_12_hour(end_hour, end_minute)

    # Determine how many days later
    day_text = ""
    if days_later == 1:
        day_text = " (next day)"
    elif days_later > 1:
        day_text = f" ({days_later} days later)"

    # Calculate the new day if the starting day was given
    if day_of_week:
        new_day = calculate_new_day(day_of_week, days_later)
        return f"{new_time}, {new_day}{day_text}"
    else:
        return f"{new_time}{day_text}"
```

## How to Use

1. **Input**: Provide a start time in `HH:MM AM/PM` format, a duration in `HH:MM` format, and an optional day of the week.
2. **Output**: Returns the end time and, if applicable, the day of the week and how many days later.

## Example Usage

```python
# Example Tests
print(add_time('3:00 PM', '3:10'))            # Expected Output: 6:10 PM
print(add_time('11:30 AM', '2:32', 'Monday')) # Expected Output: 2:02 PM, Monday
print(add_time('11:43 AM', '00:20'))          # Expected Output: 12:03 PM
print(add_time('10:10 PM', '3:30'))           # Expected Output: 1:40 AM (next day)
print(add_time('11:43 PM', '24:20', 'tueSday')) # Expected Output: 12:03 AM, Thursday (2 days later)
print(add_time('6:30 PM', '205:12'))          # Expected Output: 7:42 AM (9 days later)
```

This function can handle different time formats and correctly outputs the end time in the `HH:MM AM/PM` format, accommodating for AM/PM transitions and day changes.
