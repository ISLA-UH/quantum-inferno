"""
Top level common members and functions
"""

import quantum_inferno.utilities.errors as er

qi_debugger = er.QuantumInfernoDebugger()

def enable_debugger():
    qi_debugger.debug_mode = True
