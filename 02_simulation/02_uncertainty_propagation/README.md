## Model Implementations

Reference implementations for the *System* and *Behavior* models to produce the posteriors from observations.

### Data

To use observations for the prior construction the data needs to be a dataframe of type `Query`.

```python
class Paragraph():
    document_id: int
    index: int
    global_id: Optional[int] = None
    text: Optional[str] = None

class Query():
    model_config = ConfigDict(strict=True)
    id: str
    input: str
    answer: Optional[str] = None
    reference: list[Paragraph] = []
    retrieved: list[Paragraph] = []
    generated_answer: Optional[str] = None

    retriever_success: Optional[bool] = None
    abstain: Optional[bool] = None
    task_success: Optional[bool] = None
    generator_success: Optional[bool] = None
```

### Usage

Usage for the system model.

```python
import pandas as pd
from 01_system_model import SimulateSystemModel

observations = pd.DataFrame(obs)
model = SimulateSystemModel()

success_rates, raw_samples = model.compute_uncertainty(observations, num_samples=10_000)
```

And the same for the behavior model.

```python
import pandas as pd
from 02_behavior_model import SimulateSystemModel

observations = pd.DataFrame(obs)
model = SimulateSystemModel()

success_rates, raw_samples = model.compute_uncertainty(observations, num_samples=10_000)
```

The returned `success_rate` dict contains the means and standard deviations for all the observed marginals and conditionals.

```python
{
    'a': {'mean': 0.3075098693370819, 'std': 0.0046197399497032166},
    'a_r0': {'mean': 0.4389536380767822, 'std': 0.006162139121443033},
    'a_r1': {'mean': 0.06116431578993797, 'std': 0.004093899857252836},
    'g': {'mean': 0.4506107270717621, 'std': 0.0049493201076984406},
    'r': {'mean': 0.34792956709861755, 'std': 0.004769022576510906},
    't': {'mean': 0.26589682698249817, 'std': 0.004423532634973526},
    't_r0_a0': {'mean': 0.27694615721702576, 'std': 0.007401127368211746},
    't_r0_a1': {'mean': 0.0003405264869797975, 'std': 0.00033503244048915803},
    't_r1_a0': {'mean': 0.5032387375831604, 'std': 0.008778365328907967},
    't_r1_a1': {'mean': 0.004681094083935022, 'std': 0.0047019473277032375}
 }
 ```

