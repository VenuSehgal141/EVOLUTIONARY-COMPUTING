Packing EA
===========

This project contains a standalone evolutionary algorithm for the 2D
cylindrical container loading (circle packing) problem.

Files
- `packing_ea.py` : standalone module that implements Circle, Bunch, PackingEA, and an example runner.
- `requirements.txt` : python packages required (numpy, matplotlib)

Run
In PowerShell, from the workspace folder:

```powershell
pip install -r .\requirements.txt
python .\packing_ea.py
```

The script will run a short GA search and save a visualization `best_packing.png`.