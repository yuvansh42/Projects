
# Arithmetic Arranger

The **Arithmetic Arranger** function takes a list of arithmetic problems (addition and subtraction only) and formats them into a neatly arranged, vertically aligned structure. The function also has an optional parameter to display the answers.

## Function Code

```python
def arithmetic_arranger(problems, display_answers=False):
    # Check if there are more than 5 problems
    if len(problems) > 5:
        return 'Error: Too many problems.'

    first_line = []
    second_line = []
    dashes = []
    answers = []

    for problem in problems:
        parts = problem.split()

        # Check if the operator is either '+' or '-'
        if parts[1] not in ['+', '-']:
            return "Error: Operator must be '+' or '-'."

        # Check if both operands are digits
        if not (parts[0].isdigit() and parts[2].isdigit()):
            return 'Error: Numbers must only contain digits.'

        # Check if operands are more than four digits
        if len(parts[0]) > 4 or len(parts[2]) > 4:
            return 'Error: Numbers cannot be more than four digits.'

        # Find the width of the problem (based on the longest number)
        width = max(len(parts[0]), len(parts[2])) + 2

        # Arrange the first and second lines
        first_line.append(parts[0].rjust(width))
        second_line.append(parts[1] + parts[2].rjust(width - 1))
        dashes.append('-' * width)

        # If display_answers is True, calculate the result
        if display_answers:
            if parts[1] == '+':
                result = str(int(parts[0]) + int(parts[2]))
            else:
                result = str(int(parts[0]) - int(parts[2]))
            answers.append(result.rjust(width))

    # Join the lines with four spaces between each problem
    arranged_problems = '    '.join(first_line) + '\n' + '    '.join(second_line) + '\n' + '    '.join(dashes)
    
    if display_answers:
        arranged_problems += '\n' + '    '.join(answers)

    return arranged_problems
```

## How to Use

1. **Input**: Provide a list of arithmetic problems (e.g., `["32 + 8", "1 - 3801", "9999 + 9999", "523 - 49"]`).
2. **Optional Parameter**: Set `display_answers=True` if you want the answers displayed.

## Rules and Constraints

- Accepts a maximum of 5 problems. If more are provided, it returns an error: `'Error: Too many problems.'`
- Operands can only contain digits, and only `+` and `-` operations are allowed.
- Each operand can be a maximum of 4 digits. If more, it returns an error: `'Error: Numbers cannot be more than four digits.'`

## Example Usage

```python
problems = ["32 + 8", "1 - 3801", "9999 + 9999", "523 - 49"]
print(arithmetic_arranger(problems, display_answers=True))
```

This will display:

```
   32         1      9999      523
+   8    - 3801    + 9999    -  49
----    ------    ------    ----
  40     -3800     19998     474
```
