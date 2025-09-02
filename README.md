# MallowsPO
We include necessary components needed for implmenting ICLR 2025 paper [MallowsPO](https://arxiv.org/abs/2405.14953) in the trainer script, modified from [trl](https://github.com/huggingface/trl) DPO trainer. MallowsPO can be implemented pretty easily by modifying from the common LLM RLHF libaries' implementations on DPO. A more detailed codebase, which takes MallowsPO as a special instance (contextual scaling), can be found at [RainbowPO](https://github.com/CapitalOne-Research/RainbowPO).

## 📜 Citation

If you find **MallowsPO** useful in your research, please consider citing our work! 🚀

### BibTeX
```bibtex
@article{chen2024mallows,
  title={Mallows-DPO: Fine-Tune Your LLM with Preference Dispersions},
  author={Chen, Haoxian and Zhao, Hanyang and Lam, Henry and Yao, David and Tang, Wenpin},
  journal={arXiv preprint arXiv:2405.14953},
  year={2024}
}
