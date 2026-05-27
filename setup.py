from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="kskt",
    version="0.1.0",
    description="Know Thyself, Know Thy User: Intrinsic Dual-Perspective Reasoning for Role-Playing LLMs (ICML 2026)",
    author="Haotong Sun, Jianye Xie, Bocheng Xu, Yinghui Jiang",
    url="https://github.com/Sunrich-HT/KSKT",
    python_requires=">=3.10",
    packages=find_packages(include=["kskt", "kskt.*"]),
    install_requires=reqs,
    license="Apache-2.0",
)
