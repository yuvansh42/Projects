
# Budget Tracker

This project provides a simple budget tracking tool in Python, implemented with a `Category` class to handle deposits, withdrawals, and fund transfers across various categories. Additionally, it includes a `create_spend_chart` function that visualizes spending as a bar chart.

## Features

- **Category Class**: Tracks transactions for specific budget categories, allowing for deposits, withdrawals, and transfers.
- **Spend Chart**: A bar chart showing the percentage of spending across categories.

## Usage

1. **Category Class**
   - Create an instance with the category name.
   - Use `.deposit(amount, description)` to add funds.
   - Use `.withdraw(amount, description)` to spend funds (if available).
   - Use `.transfer(amount, category)` to move funds between categories.

2. **Create Spend Chart**
   - Pass a list of categories to `create_spend_chart` to generate a bar chart.

### Example

```python
# Create instances of categories
food = Category('Food')
clothing = Category('Clothing')
auto = Category('Auto')

# Add transactions
food.deposit(1000, 'deposit')
food.withdraw(10.15, 'groceries')
food.withdraw(15.89, 'restaurant and more food for dessert')
food.transfer(50, clothing)

auto.deposit(1000, 'initial deposit')
auto.withdraw(15)

# Print category balances
print(food)
print(clothing)
print(auto)

# Display spend chart
print(create_spend_chart([food, clothing, auto]))
```

### Output

The above code produces output showing each category's ledger with a title, transactions, and total balance. Additionally, it generates a spend chart showing the percentage of expenses per category.

## License

This project is open-source and available under the MIT License.
