
# Hat Drawing Probability Experiment

This Python script simulates drawing balls from a hat and calculates the probability of drawing a specific combination of colors, based on multiple experiments.

## Code

```python
import random

class Hat:
    def __init__(self, **kwargs):
        self.contents = []
        for color, quantity in kwargs.items():
            self.contents.extend([color] * quantity)

    def draw(self, num_balls):
        if num_balls >= len(self.contents):
            # If the number of balls to draw exceeds available balls, return all balls
            drawn_balls = self.contents.copy()  # Make a copy of all contents
            self.contents.clear()  # Empty the contents
            return drawn_balls
        
        drawn_balls = random.sample(self.contents, num_balls)
        for ball in drawn_balls:
            self.contents.remove(ball)
        
        return drawn_balls


def experiment(hat, expected_balls, num_balls_drawn, num_experiments):
    success_count = 0

    for _ in range(num_experiments):
        hat_copy = Hat(**{color: hat.contents.count(color) for color in hat.contents})
        drawn_balls = hat_copy.draw(num_balls_drawn)
        
        ball_counts = {color: drawn_balls.count(color) for color in drawn_balls}
        
        # Check if drawn balls meet the expected counts
        if all(ball_counts.get(color, 0) >= count for color, count in expected_balls.items()):
            success_count += 1

    return success_count / num_experiments
```

## Example Usage

```python
# Create a hat with specified ball quantities
hat = Hat(black=6, red=4, green=3)

# Run the experiment to find the probability of drawing at least 2 red balls and 1 green ball
probability = experiment(hat=hat,
                         expected_balls={'red': 2, 'green': 1},
                         num_balls_drawn=5,
                         num_experiments=2000)

print(probability)  # Output: Probability of drawing the expected balls
```
