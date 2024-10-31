
# Cash Register

This project is a **Cash Register** web application that calculates the change due based on cash provided by the customer and the available cash in the register. It also checks if there is sufficient cash in the drawer to complete the transaction.

## HTML Code

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cash Register</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Cash Register</h1>
        <div class="register-container">
            <div class="input-group">
                <label for="cash">Cash Provided:</label>
                <input type="number" id="cash" step="0.01" min="0">
            </div>
            <button id="purchase-btn">Make Purchase</button>
            <div id="change-due"></div>
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

## CSS Code

```css
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}

.container {
    max-width: 600px;
    margin: 0 auto;
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #333;
}

.register-container {
    margin-top: 20px;
}

.input-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    color: #555;
}

input {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-sizing: border-box;
}

button {
    width: 100%;
    padding: 10px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    margin-bottom: 15px;
}

button:hover {
    background-color: #45a049;
}

#change-due {
    padding: 10px;
    background-color: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 4px;
    min-height: 20px;
}
```

## JavaScript Code

```javascript
let price = 19.5;
let cid = [
    ["PENNY", 1.01],
    ["NICKEL", 2.05],
    ["DIME", 3.1],
    ["QUARTER", 4.25],
    ["ONE", 90],
    ["FIVE", 55],
    ["TEN", 20],
    ["TWENTY", 60],
    ["ONE HUNDRED", 100]
];

const CURRENCY_UNIT = {
    "ONE HUNDRED": 100,
    "TWENTY": 20,
    "TEN": 10,
    "FIVE": 5,
    "ONE": 1,
    "QUARTER": 0.25,
    "DIME": 0.1,
    "NICKEL": 0.05,
    "PENNY": 0.01
};

document.getElementById("purchase-btn").addEventListener("click", () => {
    const cashInput = parseFloat(document.getElementById("cash").value);
    const changeDueElement = document.getElementById("change-due");
    
    // Check if customer has enough money
    if (cashInput < price) {
        alert("Customer does not have enough money to purchase the item");
        return;
    }

    // If exact change, no need to calculate
    if (cashInput === price) {
        changeDueElement.textContent = "No change due - customer paid with exact cash";
        return;
    }

    let changeDue = Math.round((cashInput - price) * 100) / 100;
    const totalCID = cid.reduce((sum, [_, amount]) => sum + amount, 0);

    // Clone the cid array to avoid modifying the original
    let availableCID = cid.map(([unit, amount]) => [unit, amount]);

    // If exact change in drawer
    if (totalCID === changeDue) {
        let changeString = "Status: CLOSED";
        for (let [unit, amount] of availableCID) {
            if (amount > 0) {
                changeString += ` ${unit}: $${amount}`;
            }
        }
        changeDueElement.textContent = changeString;
        return;
    }

    // Check if we have enough money in the drawer
    if (totalCID < changeDue) {
        changeDueElement.textContent = "Status: INSUFFICIENT_FUNDS";
        return;
    }

    // Calculate change breakdown
    let changeArray = [];
    let remainingChange = changeDue;

    // Convert to cents to avoid floating point issues
    remainingChange = Math.round(remainingChange * 100);

    for (let unit in CURRENCY_UNIT) {
        const unitValue = Math.round(CURRENCY_UNIT[unit] * 100);
        const availableUnit = availableCID.find(item => item[0] === unit);
        
        if (!availableUnit) continue;

        let availableAmount = Math.round(availableUnit[1] * 100);
        let unitCount = 0;

        while (remainingChange >= unitValue && availableAmount >= unitValue) {
            remainingChange -= unitValue;
            availableAmount -= unitValue;
            unitCount += unitValue;
        }

        if (unitCount > 0) {
            changeArray.push([unit, unitCount / 100]);
        }
    }

    // Check if we could make exact change
    if (remainingChange > 0) {
        changeDueElement.textContent = "Status: INSUFFICIENT_FUNDS";
        return;
    }

    // Format and display change
    let changeString = "Status: OPEN";
    changeArray.forEach(([unit, amount]) => {
        if (amount > 0) {
            changeString += ` ${unit}: $${amount}`;
        }
    });
    
    changeDueElement.textContent = changeString;
});
```

---

This README file includes the complete HTML, CSS, and JavaScript code blocks for the Cash Register project.
