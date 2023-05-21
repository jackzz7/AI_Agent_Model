#include"python_API.h"

namespace emb
{
    struct Stdout
    {
        PyObject_HEAD
            stdout_write_type write;
    };
    struct Stdin
    {
        PyObject_HEAD
            stdin_read_type read;
    };

    PyObject* Stdin_readline(PyObject* self, PyObject* args)
    {
        Stdin* selfimpl = reinterpret_cast<Stdin*>(self);
        std::string str("");
        if (selfimpl->read)
        {
            selfimpl->read(str);
        }
        else assert(false);
        return Py_BuildValue("s", str.c_str());
    }
    PyObject* Stdout_write(PyObject* self, PyObject* args)
    {
        std::size_t written(0);
        Stdout* selfimpl = reinterpret_cast<Stdout*>(self);
        if (selfimpl->write)
        {
            char* data;
            if (!PyArg_ParseTuple(args, "s", &data))
                return 0;

            std::string str(data);
            selfimpl->write(str);
            written = str.size();
        }
        return PyLong_FromSize_t(written);
    }

    PyObject* Stdout_flush(PyObject* self, PyObject* args)
    {
        // no-op
        return Py_BuildValue("");
    }
    PyObject* Stdout_fileno(PyObject* self, PyObject* args)
    {
        // no-op
        return Py_BuildValue("i", 1);
    }

    PyMethodDef Stdout_methods[] =
    {
        {"write", Stdout_write, METH_VARARGS, "sys.stdout.in"},
        {"flush", Stdout_flush, METH_VARARGS, "sys.stdout.flush"},
        {"fileno", Stdout_fileno, METH_VARARGS, "sys.stdout.fileno"},
        {0, 0, 0, 0} // sentinel
    };
    PyMethodDef Stdin_methods[] =
    {
        {"readline", Stdin_readline, METH_VARARGS, "sys.stdin.readline"},
        {0, 0, 0, 0} // sentinel
    };

    PyTypeObject StdoutType =
    {
        PyVarObject_HEAD_INIT(0, 0)
        "emb.StdoutType",     /* tp_name */
        sizeof(Stdout),       /* tp_basicsize */
        0,                    /* tp_itemsize */
        0,                    /* tp_dealloc */
        0,                    /* tp_print */
        0,                    /* tp_getattr */
        0,                    /* tp_setattr */
        0,                    /* tp_reserved */
        0,                    /* tp_repr */
        0,                    /* tp_as_number */
        0,                    /* tp_as_sequence */
        0,                    /* tp_as_mapping */
        0,                    /* tp_hash  */
        0,                    /* tp_call */
        0,                    /* tp_str */
        0,                    /* tp_getattro */
        0,                    /* tp_setattro */
        0,                    /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,   /* tp_flags */
        "emb.Stdout objects", /* tp_doc */
        0,                    /* tp_traverse */
        0,                    /* tp_clear */
        0,                    /* tp_richcompare */
        0,                    /* tp_weaklistoffset */
        0,                    /* tp_iter */
        0,                    /* tp_iternext */
        Stdout_methods,       /* tp_methods */
        0,                    /* tp_members */
        0,                    /* tp_getset */
        0,                    /* tp_base */
        0,                    /* tp_dict */
        0,                    /* tp_descr_get */
        0,                    /* tp_descr_set */
        0,                    /* tp_dictoffset */
        0,                    /* tp_init */
        0,                    /* tp_alloc */
        0,                    /* tp_new */
    };
    PyTypeObject StdinType =
    {
        PyVarObject_HEAD_INIT(0, 0)
        "emb.StdinType",     /* tp_name */
        sizeof(Stdin),       /* tp_basicsize */
        0,                    /* tp_itemsize */
        0,                    /* tp_dealloc */
        0,                    /* tp_print */
        0,                    /* tp_getattr */
        0,                    /* tp_setattr */
        0,                    /* tp_reserved */
        0,                    /* tp_repr */
        0,                    /* tp_as_number */
        0,                    /* tp_as_sequence */
        0,                    /* tp_as_mapping */
        0,                    /* tp_hash  */
        0,                    /* tp_call */
        0,                    /* tp_str */
        0,                    /* tp_getattro */
        0,                    /* tp_setattro */
        0,                    /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,   /* tp_flags */
        "emb.Stdin objects", /* tp_doc */
        0,                    /* tp_traverse */
        0,                    /* tp_clear */
        0,                    /* tp_richcompare */
        0,                    /* tp_weaklistoffset */
        0,                    /* tp_iter */
        0,                    /* tp_iternext */
        Stdin_methods,       /* tp_methods */
        0,                    /* tp_members */
        0,                    /* tp_getset */
        0,                    /* tp_base */
        0,                    /* tp_dict */
        0,                    /* tp_descr_get */
        0,                    /* tp_descr_set */
        0,                    /* tp_dictoffset */
        0,                    /* tp_init */
        0,                    /* tp_alloc */
        0,                    /* tp_new */
    };

    PyModuleDef embmodule =
    {
        PyModuleDef_HEAD_INIT,
        "emb", 0, -1, 0,
    };

    // Internal state
    PyObject* g_stdout, * g_stdin;
    PyObject* g_stdout_saved, * g_stdin_saved;

    PyMODINIT_FUNC PyInit_emb(void)
    {
        g_stdin = g_stdout = 0;
        g_stdin_saved = g_stdout_saved = 0;

        StdoutType.tp_new = PyType_GenericNew;
        StdinType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&StdoutType) < 0 || PyType_Ready(&StdinType) < 0)
            return 0;

        PyObject* m = PyModule_Create(&embmodule);
        if (m)
        {
            Py_INCREF(&StdoutType);
            Py_INCREF(&StdinType);
            PyModule_AddObject(m, "Stdout", reinterpret_cast<PyObject*>(&StdoutType));
            PyModule_AddObject(m, "Stdin", reinterpret_cast<PyObject*>(&StdinType));
        }
        return m;
    }

    void set_stdout_and_stdin(stdout_write_type write, stdin_read_type read)
    {
        if (!g_stdout)
        {
            g_stdout_saved = PySys_GetObject("stdout"); // borrowed
            g_stdout = StdoutType.tp_new(&StdoutType, 0, 0);
            g_stdin_saved = PySys_GetObject("stdin"); // borrowed
            g_stdin = StdinType.tp_new(&StdinType, 0, 0);
        }

        reinterpret_cast<Stdout*>(g_stdout)->write = write;
        reinterpret_cast<Stdin*>(g_stdin)->read = read;
        PySys_SetObject("stdin", g_stdin);
        PySys_SetObject("stdout", g_stdout);
    }

    void reset_stdout_and_stdin()
    {
        if (g_stdout_saved)
            PySys_SetObject("stdout", g_stdout_saved), PySys_SetObject("stdin", g_stdin_saved);

        Py_XDECREF(g_stdout);
        Py_XDECREF(g_stdin);
        g_stdin = g_stdout = 0;
    }

} // namespace emb