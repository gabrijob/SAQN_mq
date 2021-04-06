#include "Python.h" 
//#include "AgentAPI.h"
#include "SAQNAgent.h"

int main(int argc, char *argv[]) {
    
    PyObject *pmodule;
    wchar_t *program;
    
    program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0], got %d arguments\n", argc);
        exit(1);
    }
    
    /* Add a built-in module, before Py_Initialize */    
    if (PyImport_AppendInittab("SAQNAgent", PyInit_SAQNAgent) == -1) {   
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }
    
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */ 
    pmodule = PyImport_ImportModule("SAQNAgent");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'SAQNAgent'\n");
        goto exit_with_error;
    }

    /* Now call into your module code. */
    printf("\nStarting Agent");
    float start_state[6] = {0.0,0.0,0.0,0,0,0};
    PyObject* agent = createAgent(start_state, 30, 4);

    printf("\nGetting first action");
    float middle_state[6] = {30.0, 5.3, 5.3, 10, 8, 8};
    int act = infer(agent, middle_state);
    printf("\nAction is %d", act);

    printf("\nClosing Agent");
    float last_state[6] = {8000.0, 1.1, 1.1, 10, 8, 8};
    finish(agent, last_state);

    /* Clean up after using CPython. */
    PyMem_RawFree(program);
    Py_Finalize();

    return 0;

    /* Clean up in the error cases above. */

exit_with_error:
    PyMem_RawFree(program);
    Py_Finalize();
    return 1;

}
