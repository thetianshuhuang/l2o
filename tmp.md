### Best
- Soft Choice: MLP (No Random Scaling)
- RNNProp: Conv + IL (With Repeat)

### Good
- Soft Choice: MLP
- Soft Choice: MLP, 2x iterations

### Slow/Stalled
- Soft Choice: Conv
- Soft Choice: Conv, 2x iterations
- Hard Choice: Conv
- Soft Choice: Conv (No Random Scaling)
- RNNProp: MLP
- RNNProp: Conv + 10x IL
- RNNProp: MLP + 10x IL
- RNNProp: MLP + IL (No Random Scaling)
- RNNProp: Conv + No Annealing IL @ 20%
- RNNProp: Conv + IL (With Repeat) @ 20%

### Loss Exploded
- Hard Choice: MLP
- RNNProp: Conv (With Repeat)
- RNNProp: Conv + No Annealing IL
- RNNProp: Conv + Slower Annealing IL
- RNNProp: Conv + IL (No Random Scaling)
- RNNProp: MLP + IL (With Repeat)
- RNNProp: MLP + Slower Annealing IL
- RNNProp: MLP + No Annealing IL

### Failed
- RNNProp: Conv
- RNNProp: MLP (with repeat)
