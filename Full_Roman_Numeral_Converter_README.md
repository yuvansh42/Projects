
# Roman Numeral Converter

This repository contains a simple web application that converts numbers (1–3999) to Roman numerals.

## HTML Code

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Roman Numeral Converter</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Roman Numeral Converter</h1>
        <div class="converter-box">
            <div class="input-group">
                <input type="number" id="number" placeholder="Enter a number (1-3999)">
                <button id="convert-btn">Convert</button>
            </div>
            <div id="output" class="output"></div>
        </div>
        <div class="reference">
            <h2>Quick Reference</h2>
            <div class="grid">
                <div>M = 1000</div>
                <div>D = 500</div>
                <div>C = 100</div>
                <div>L = 50</div>
                <div>X = 10</div>
                <div>V = 5</div>
                <div>I = 1</div>
            </div>
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

## CSS Code

```css
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f2f5;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    max-width: 600px;
    width: 100%;
    padding: 20px;
}

h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
}

.converter-box {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.input-group {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}

#number {
    flex: 1;
    padding: 12px;
    font-size: 16px;
    border: 2px solid #ddd;
    border-radius: 5px;
    outline: none;
}

#number:focus {
    border-color: #3498db;
}

#convert-btn {
    padding: 12px 24px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.2s;
}

#convert-btn:hover {
    background-color: #2980b9;
}

.output {
    min-height: 40px;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 5px;
    font-size: 18px;
    text-align: center;
}

.reference {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.reference h2 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 15px;
    font-size: 1.2em;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 10px;
    text-align: center;
}

.grid div {
    background-color: #f8f9fa;
    padding: 8px;
    border-radius: 5px;
    font-family: monospace;
}
```

## JavaScript Code

```javascript
const romanNumerals = [
    { value: 1000, numeral: 'M' },
    { value: 900, numeral: 'CM' },
    { value: 500, numeral: 'D' },
    { value: 400, numeral: 'CD' },
    { value: 100, numeral: 'C' },
    { value: 90, numeral: 'XC' },
    { value: 50, numeral: 'L' },
    { value: 40, numeral: 'XL' },
    { value: 10, numeral: 'X' },
    { value: 9, numeral: 'IX' },
    { value: 5, numeral: 'V' },
    { value: 4, numeral: 'IV' },
    { value: 1, numeral: 'I' }
];

function convertToRoman(num) {
    let result = '';
    let remainingNum = num;
    
    for (let { value, numeral } of romanNumerals) {
        while (remainingNum >= value) {
            result += numeral;
            remainingNum -= value;
        }
    }
    
    return result;
}

function validateAndConvert() {
    const numberInput = document.getElementById('number');
    const output = document.getElementById('output');
    const num = numberInput.value;

    // Check if input is empty
    if (!num) {
        output.textContent = 'Please enter a valid number';
        return;
    }

    // Convert to number and validate
    const number = parseInt(num);
    
    if (number < 1) {
        output.textContent = 'Please enter a number greater than or equal to 1';
        return;
    }
    
    if (number >= 4000) {
        output.textContent = 'Please enter a number less than or equal to 3999';
        return;
    }

    // Convert and display result
    const romanNumeral = convertToRoman(number);
    output.textContent = romanNumeral;
}

// Add event listeners
document.getElementById('convert-btn').addEventListener('click', validateAndConvert);
document.getElementById('number').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        validateAndConvert();
    }
});
```

--- 

This README includes the full HTML, CSS, and JavaScript code blocks for the Roman Numeral Converter project.
