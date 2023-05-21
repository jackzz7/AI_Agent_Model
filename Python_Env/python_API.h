#pragma once
#include <functional>
#include <iostream>
#include <string>

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

namespace emb
{

    typedef std::function<void(const std::string&)> stdout_write_type;
    typedef std::function<void(std::string&)> stdin_read_type;

    //PyObject* Stdout_readline(PyObject* self, PyObject* args);

    //PyObject* Stdout_write(PyObject* self, PyObject* args);

    //PyObject* Stdout_flush(PyObject* self, PyObject* args);

    //extern PyMethodDef Stdout_methods[];

    //extern PyTypeObject StdoutType, StdinType;

    //extern PyModuleDef embmodule;

    // Internal state
    //extern PyObject* g_stdout;
    //extern PyObject* g_stdout_saved;

    PyMODINIT_FUNC PyInit_emb(void);

    void set_stdout_and_stdin(stdout_write_type write, stdin_read_type read);

    void reset_stdout_and_stdin();

} // namespace emb