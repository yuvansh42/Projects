
# Survey Form

This repository contains a basic survey form designed to gather user feedback on various service aspects.

## HTML Code

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Form</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1 id="title">Survey Form</h1>
    <p id="description">We would love to hear your feedback. Please fill out the survey below.</p>

    <form id="survey-form">
        <!-- Name input -->
        <label id="name-label" for="name">Name:</label>
        <input type="text" id="name" name="name" placeholder="Enter your name" required>

        <!-- Email input -->
        <label id="email-label" for="email">Email:</label>
        <input type="email" id="email" name="email" placeholder="Enter your email" required>

        <!-- Number input -->
        <label id="number-label" for="number">Age:</label>
        <input type="number" id="number" name="age" min="18" max="100" placeholder="Enter your age" required>

        <!-- Dropdown select -->
        <label for="dropdown">Which option best describes you?</label>
        <select id="dropdown" name="role" required>
            <option value="" disabled selected>Select an option</option>
            <option value="student">Student</option>
            <option value="full-time-job">Full-time job</option>
            <option value="self-employed">Self-employed</option>
            <option value="unemployed">Unemployed</option>
        </select>

        <!-- Radio buttons -->
        <p>How did you hear about us?</p>
        <label>
            <input type="radio" name="source" value="friends" required> Friends
        </label>
        <label>
            <input type="radio" name="source" value="social-media" required> Social Media
        </label>
        <label>
            <input type="radio" name="source" value="other" required> Other
        </label>

        <!-- Checkboxes -->
        <p>What do you like about our service? (Select all that apply)</p>
        <label>
            <input type="checkbox" name="feature" value="quality"> Quality
        </label>
        <label>
            <input type="checkbox" name="feature" value="price"> Price
        </label>
        <label>
            <input type="checkbox" name="feature" value="customer-service"> Customer Service
        </label>

        <!-- Textarea for comments -->
        <label for="comments">Any comments or suggestions?</label>
        <textarea id="comments" name="comments" rows="4" cols="50" placeholder="Enter your comments here..."></textarea>

        <!-- Submit button -->
        <button type="submit" id="submit">Submit</button>
    </form>
</body>
</html>
```
