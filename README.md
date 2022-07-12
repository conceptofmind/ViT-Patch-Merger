## Patch Merger

<img src="./images/patch_merger.png" width="400px"></img>

"Transformers are widely applied to solve natural language understanding and computer vision tasks. While scaling up these architectures leads to improved performance, it often comes at the expense of much higher computational costs. In order for large-scale models to remain practical in real-world systems, there is a need for reducing their computational overhead. In this work, we present the PatchMerger, a simple module that reduces the number of patches or tokens the network has to process by merging them between two consecutive intermediate layers. We show that the PatchMerger achieves a significant speedup across various model sizes while matching the original performance both upstream and downstream after fine-tuning." - Cedric Renggli, André Susano Pinto, Neil Houlsby, Basil Mustafa, Joan Puigcerver, Carlos Riquelme

### Research Paper:
- https://arxiv.org/abs/2202.12015

### Usage:
```python
import numpy as np

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (1, 256, 256, 3))

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 12,
    heads = 8,
    patch_merge_layer = 6,        # at which transformer layer to do patch merging
    patch_merge_num_tokens = 8,   # the output number of tokens from the patch merge
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")

key = jax.random.PRNGKey(5)

features = jax.random.normal(key, (4, 256, 1024))

merger = PatchMerger(
    dim = 1024,
    num_tokens_out = 8   # output number of tokens
)

merger_params = merger.init(init_rngs, features)
merger_output = merger.apply(merger_params, features)
print(merger_output.shape)
```
## Citation:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2202.12015,
  doi = {10.48550/ARXIV.2202.12015},
  
  url = {https://arxiv.org/abs/2202.12015},
  
  author = {Renggli, Cedric and Pinto, André Susano and Houlsby, Neil and Mustafa, Basil and Puigcerver, Joan and Riquelme, Carlos},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Learning to Merge Tokens in Vision Transformers},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```