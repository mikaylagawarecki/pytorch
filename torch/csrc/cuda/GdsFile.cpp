#include <pybind11/pybind11.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include <ATen/cuda/CUDAGdsFile.h>
#include <structmember.h>

PyObject* THCPGdsFileClass = nullptr;

static PyObject* THCPGdsFile_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // NOLINTNEXTLINE(*-c-arrays*)
  constexpr const char* kwlist[] = {"filename", "mode", nullptr};
  const char* filename = nullptr;
  const char* mode = nullptr;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "ss",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &filename,
          &mode)) {
    return nullptr;
  }

  // TODO: Need error checking for filename and mode?

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THCPGdsFile* self = (THCPGdsFile*)ptr.get();

  new (&self->gds_file) at::cuda::GDSFile(filename, mode);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPGdsFile_dealloc(THCPGdsFile* self) {
  self->gds_file.~GDSFile();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THCPGdsFile_load_data(PyObject* _self, PyObject* t_) {
  HANDLE_TH_ERRORS
  auto self = (THCPGdsFile*)_self;
  TORCH_CHECK(THPVariable_Check(t_));
  auto& t = THPVariable_Unpack(t_);
  self->gds_file.load_data(t);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPGdsFile_save_data(PyObject* _self, PyObject* t_) {
  HANDLE_TH_ERRORS
  auto self = (THCPGdsFile*)_self;
  TORCH_CHECK(THPVariable_Check(t_));
  auto& t = THPVariable_Unpack(t_);
  self->gds_file.save_data(t);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static struct PyGetSetDef THCPGdsFile_properties[] = {
    // FIXME: add properties (perhaps getter for filename)
    // {"filename", (getter)THCPGdsFile_get_filename, nullptr, nullptr,
    // nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static PyMethodDef THCPGdsFile_methods[] = {
    {(char*)"load_data", THCPGdsFile_load_data, METH_O, nullptr},
    {(char*)"save_data", THCPGdsFile_save_data, METH_O, nullptr},
    {nullptr}};

PyTypeObject THCPGdsFileType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._CudaGdsFileBase", /* tp_name */
    sizeof(THCPGdsFile), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THCPGdsFile_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THCPGdsFile_methods, /* tp_methods */
    nullptr, /* tp_members */
    THCPGdsFile_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THCPGdsFile_pynew, /* tp_new */
};

void THCPGdsFile_init(PyObject* module) {
  THCPGdsFileClass = (PyObject*)&THCPGdsFileType;
  if (PyType_Ready(&THCPGdsFileType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPGdsFileType);
  if (PyModule_AddObject(
          module, "_CudaGdsFileBase", (PyObject*)&THCPGdsFileType) < 0) {
    throw python_error();
  }
}
