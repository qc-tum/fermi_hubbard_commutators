from setuptools import setup


setup(
    name="fh_comm",
    version="1.0.0",
    author="Christian B. Mendl",
    author_email="christian.b.mendl@gmail.com",
    packages=["fh_comm"],
    url="https://github.com/qc-tum/fermi_hubbard_commutators",
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
    ],
)
