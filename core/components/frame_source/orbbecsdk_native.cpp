#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "libobsensor/ObSensor.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ReaderConfig {
    std::string sdkRoot;
    int width = 640;
    int height = 480;
    int fps = 30;
    std::string alignMode = "hardware";
    bool enableFrameSync = false;
    double approxSyncMs = 50.0;
    int maxQueueSize = 10;
    int waitTimeoutMs = 1000;
};

struct CopiedFrame {
    uint64_t index = 0;
    uint64_t deviceTsUs = 0;
    uint64_t systemTsUs = 0;
    uint64_t globalTsUs = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    OBFormat format = OB_FORMAT_UNKNOWN;
    std::vector<uint8_t> bytes;
};

struct MatchedSample {
    CopiedFrame color;
    CopiedFrame depth;
    int64_t deltaSystemUs = 0;
};

std::string joinPath(const std::string &lhs, const std::string &rhs) {
    if(lhs.empty()) {
        return rhs;
    }
    if(lhs[lhs.size() - 1] == '/') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

std::string makeConfigPath(const std::string &sdkRoot) {
    if(sdkRoot.empty()) {
        return "";
    }
    return joinPath(sdkRoot, "config/OrbbecSDKConfig_v1.0.xml");
}

std::shared_ptr<ob::VideoStreamProfile> chooseColorProfile(
    ob::Pipeline &pipeline,
    int width,
    int height,
    int fps
) {
    auto profiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    try {
        return profiles->getVideoStreamProfile(width, height, OB_FORMAT_RGB, fps);
    }
    catch(...) {
    }
    try {
        return profiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_RGB, OB_FPS_ANY);
    }
    catch(...) {
    }
    return profiles->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
}

std::shared_ptr<ob::VideoStreamProfile> chooseDepthProfile(
    ob::Pipeline &pipeline,
    const std::shared_ptr<ob::VideoStreamProfile> &colorProfile,
    OBAlignMode alignMode
) {
    auto profiles = pipeline.getD2CDepthProfileList(colorProfile, alignMode);
    if(!profiles || profiles->count() == 0) {
        return nullptr;
    }
    try {
        return profiles->getVideoStreamProfile(
            OB_WIDTH_ANY,
            OB_HEIGHT_ANY,
            OB_FORMAT_Y16,
            static_cast<int>(colorProfile->fps())
        );
    }
    catch(...) {
    }
    return profiles->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
}

int64_t signedDelta(uint64_t lhs, uint64_t rhs) {
    return static_cast<int64_t>(lhs) - static_cast<int64_t>(rhs);
}

int64_t absDelta(uint64_t lhs, uint64_t rhs) {
    int64_t value = signedDelta(lhs, rhs);
    return value < 0 ? -value : value;
}

double percentileMs(std::vector<int64_t> values, double q) {
    if(values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
    return static_cast<double>(values[index]) / 1000.0;
}

class OrbbecSdkReader {
public:
    explicit OrbbecSdkReader(const ReaderConfig &config)
        : config_(config),
          configPath_(makeConfigPath(config.sdkRoot)) {
        if(config_.approxSyncMs <= 0.0) {
            throw std::runtime_error("approxSyncMs must be > 0 for OrbbecSDK FrameSource");
        }
        thresholdUs_ = static_cast<int64_t>(config_.approxSyncMs * 1000.0);
        maxQueueSize_ = static_cast<size_t>(std::max(2, config_.maxQueueSize));

        ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
        ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_ERROR);
        context_.reset(new ob::Context(configPath_.c_str()));

        auto devices = context_->queryDeviceList();
        if(!devices || devices->deviceCount() == 0) {
            throw std::runtime_error("OrbbecSDK did not find any device");
        }

        device_ = devices->getDevice(0);
        pipeline_.reset(new ob::Pipeline(device_));
        start();
    }

    ~OrbbecSdkReader() {
        close();
    }

    MatchedSample next() {
        if(closed_) {
            throw std::runtime_error("OrbbecSdkReader is closed");
        }

        for(int attempt = 0; attempt < 50; ++attempt) {
            MatchedSample sample;
            if(match(sample)) {
                return sample;
            }

            auto frameset = pipeline_->waitForFrames(static_cast<uint32_t>(config_.waitTimeoutMs));
            if(frameset == nullptr) {
                continue;
            }

            auto color = frameset->colorFrame();
            auto depth = frameset->depthFrame();
            if(color != nullptr) {
                colors_.push_back(copyColor(color));
                ++pushedColor_;
            }
            if(depth != nullptr) {
                depths_.push_back(copyDepth(depth));
                ++pushedDepth_;
            }
            trimQueues();
        }

        MatchedSample sample;
        if(match(sample)) {
            return sample;
        }
        throw std::runtime_error("Timed out waiting for approximate-synced RGB/depth frames");
    }

    PyObject *intrinsicDict() const {
        PyObject *dict = PyDict_New();
        if(dict == nullptr) {
            return nullptr;
        }
        setLong(dict, "width", cameraParam_.rgbIntrinsic.width);
        setLong(dict, "height", cameraParam_.rgbIntrinsic.height);
        setDouble(dict, "fx", cameraParam_.rgbIntrinsic.fx);
        setDouble(dict, "fy", cameraParam_.rgbIntrinsic.fy);
        setDouble(dict, "cx", cameraParam_.rgbIntrinsic.cx);
        setDouble(dict, "cy", cameraParam_.rgbIntrinsic.cy);
        return dict;
    }

    PyObject *statsDict() const {
        PyObject *dict = PyDict_New();
        if(dict == nullptr) {
            return nullptr;
        }
        setLong(dict, "pushed_color", pushedColor_);
        setLong(dict, "pushed_depth", pushedDepth_);
        setLong(dict, "matched", matched_);
        setLong(dict, "dropped_color", droppedColor_);
        setLong(dict, "dropped_depth", droppedDepth_);
        setLong(dict, "pending_color", static_cast<long long>(colors_.size()));
        setLong(dict, "pending_depth", static_cast<long long>(depths_.size()));
        setDouble(dict, "threshold_ms", config_.approxSyncMs);
        setDouble(dict, "min_delta_ms", percentileMs(deltaUs_, 0.0));
        setDouble(dict, "p50_delta_ms", percentileMs(deltaUs_, 0.50));
        setDouble(dict, "p95_delta_ms", percentileMs(deltaUs_, 0.95));
        setDouble(dict, "max_delta_ms", percentileMs(deltaUs_, 1.0));
        return dict;
    }

    void close() {
        if(closed_) {
            return;
        }
        closed_ = true;
        if(pipeline_) {
            try {
                pipeline_->stop();
            }
            catch(...) {
            }
        }
        pipeline_.reset();
        device_.reset();
        context_.reset();
    }

private:
    ReaderConfig config_;
    std::string configPath_;
    std::unique_ptr<ob::Context> context_;
    std::shared_ptr<ob::Device> device_;
    std::unique_ptr<ob::Pipeline> pipeline_;
    OBCameraParam cameraParam_{};
    bool closed_ = false;
    bool frameSyncEnabled_ = false;
    int64_t thresholdUs_ = 50000;
    size_t maxQueueSize_ = 10;
    std::deque<CopiedFrame> colors_;
    std::deque<CopiedFrame> depths_;
    std::vector<int64_t> deltaUs_;
    int pushedColor_ = 0;
    int pushedDepth_ = 0;
    int matched_ = 0;
    int droppedColor_ = 0;
    int droppedDepth_ = 0;

    void start() {
        auto config = std::make_shared<ob::Config>();
        auto colorProfile = chooseColorProfile(*pipeline_, config_.width, config_.height, config_.fps);
        config->enableStream(colorProfile);

        OBAlignMode alignMode = ALIGN_D2C_HW_MODE;
        if(config_.alignMode == "off" || config_.alignMode == "disable") {
            alignMode = ALIGN_DISABLE;
        }
        else if(config_.alignMode == "software") {
            alignMode = ALIGN_D2C_SW_MODE;
        }

        auto depthProfile = (alignMode == ALIGN_DISABLE)
            ? pipeline_->getStreamProfileList(OB_SENSOR_DEPTH)->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>()
            : chooseDepthProfile(*pipeline_, colorProfile, alignMode);
        if(depthProfile == nullptr) {
            throw std::runtime_error("No D2C-compatible depth profile found");
        }
        config->enableStream(depthProfile);
        config->setAlignMode(alignMode);

        if(config_.enableFrameSync) {
            try {
                pipeline_->enableFrameSync();
                frameSyncEnabled_ = true;
            }
            catch(...) {
                frameSyncEnabled_ = false;
            }
        }

        pipeline_->start(config);
        cameraParam_ = pipeline_->getCameraParam();
    }

    CopiedFrame copyColor(const std::shared_ptr<ob::ColorFrame> &frame) {
        if(frame->format() != OB_FORMAT_RGB) {
            throw std::runtime_error("Color frame format is not RGB");
        }
        CopiedFrame copied;
        fillCommon(copied, frame);
        size_t expected = static_cast<size_t>(copied.width) * copied.height * 3;
        if(frame->dataSize() < expected) {
            throw std::runtime_error("Color frame data is smaller than expected RGB size");
        }
        const uint8_t *data = static_cast<const uint8_t *>(frame->data());
        copied.bytes.assign(data, data + expected);
        return copied;
    }

    CopiedFrame copyDepth(const std::shared_ptr<ob::DepthFrame> &frame) {
        CopiedFrame copied;
        fillCommon(copied, frame);
        size_t expected = static_cast<size_t>(copied.width) * copied.height * sizeof(uint16_t);
        if(frame->dataSize() < expected) {
            throw std::runtime_error("Depth frame data is smaller than expected Y16 size");
        }
        const uint8_t *data = static_cast<const uint8_t *>(frame->data());
        copied.bytes.assign(data, data + expected);
        return copied;
    }

    template <typename TFrame>
    void fillCommon(CopiedFrame &copied, const std::shared_ptr<TFrame> &frame) {
        copied.index = frame->index();
        copied.deviceTsUs = frame->timeStampUs();
        copied.systemTsUs = frame->systemTimeStampUs();
        copied.globalTsUs = frame->globalTimeStampUs();
        copied.width = frame->width();
        copied.height = frame->height();
        copied.format = frame->format();
    }

    bool match(MatchedSample &sample) {
        trimQueues();
        while(!colors_.empty() && !depths_.empty()) {
            size_t bestColor = 0;
            size_t bestDepth = 0;
            int64_t bestAbsDelta = std::numeric_limits<int64_t>::max();

            for(size_t ci = 0; ci < colors_.size(); ++ci) {
                for(size_t di = 0; di < depths_.size(); ++di) {
                    int64_t delta = absDelta(colors_[ci].systemTsUs, depths_[di].systemTsUs);
                    if(delta < bestAbsDelta) {
                        bestAbsDelta = delta;
                        bestColor = ci;
                        bestDepth = di;
                    }
                }
            }

            if(bestAbsDelta <= thresholdUs_) {
                sample.color = colors_[bestColor];
                sample.depth = depths_[bestDepth];
                sample.deltaSystemUs = signedDelta(sample.color.systemTsUs, sample.depth.systemTsUs);
                deltaUs_.push_back(bestAbsDelta);
                colors_.erase(colors_.begin() + static_cast<std::deque<CopiedFrame>::difference_type>(bestColor));
                depths_.erase(depths_.begin() + static_cast<std::deque<CopiedFrame>::difference_type>(bestDepth));
                ++matched_;
                return true;
            }

            uint64_t newestColor = colors_.back().systemTsUs;
            uint64_t newestDepth = depths_.back().systemTsUs;
            if(static_cast<int64_t>(colors_.front().systemTsUs) + thresholdUs_ < static_cast<int64_t>(newestDepth)) {
                colors_.pop_front();
                ++droppedColor_;
                continue;
            }
            if(static_cast<int64_t>(depths_.front().systemTsUs) + thresholdUs_ < static_cast<int64_t>(newestColor)) {
                depths_.pop_front();
                ++droppedDepth_;
                continue;
            }
            return false;
        }
        return false;
    }

    void trimQueues() {
        while(colors_.size() > maxQueueSize_) {
            colors_.pop_front();
            ++droppedColor_;
        }
        while(depths_.size() > maxQueueSize_) {
            depths_.pop_front();
            ++droppedDepth_;
        }
    }

    static void setLong(PyObject *dict, const char *key, long long value) {
        PyObject *object = PyLong_FromLongLong(value);
        PyDict_SetItemString(dict, key, object);
        Py_XDECREF(object);
    }

    static void setDouble(PyObject *dict, const char *key, double value) {
        PyObject *object = PyFloat_FromDouble(value);
        PyDict_SetItemString(dict, key, object);
        Py_XDECREF(object);
    }
};

typedef struct {
    PyObject_HEAD
    OrbbecSdkReader *reader;
} PyOrbbecSdkReader;

PyObject *arrayFromColor(const CopiedFrame &frame) {
    npy_intp dims[3] = {
        static_cast<npy_intp>(frame.height),
        static_cast<npy_intp>(frame.width),
        3,
    };
    PyObject *array = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if(array == nullptr) {
        return nullptr;
    }
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(array)), frame.bytes.data(), frame.bytes.size());
    return array;
}

PyObject *arrayFromDepth(const CopiedFrame &frame) {
    npy_intp dims[2] = {
        static_cast<npy_intp>(frame.height),
        static_cast<npy_intp>(frame.width),
    };
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_UINT16);
    if(array == nullptr) {
        return nullptr;
    }
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(array)), frame.bytes.data(), frame.bytes.size());
    return array;
}

int setDictItemSteal(PyObject *dict, const char *key, PyObject *value) {
    if(value == nullptr) {
        return -1;
    }
    int result = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return result;
}

PyObject *sampleToDict(const MatchedSample &sample) {
    PyObject *dict = PyDict_New();
    if(dict == nullptr) {
        return nullptr;
    }

    if(setDictItemSteal(dict, "rgb", arrayFromColor(sample.color)) < 0
       || setDictItemSteal(dict, "depth", arrayFromDepth(sample.depth)) < 0
       || setDictItemSteal(dict, "color_index", PyLong_FromUnsignedLongLong(sample.color.index)) < 0
       || setDictItemSteal(dict, "depth_index", PyLong_FromUnsignedLongLong(sample.depth.index)) < 0
       || setDictItemSteal(dict, "color_system_ts_us", PyLong_FromUnsignedLongLong(sample.color.systemTsUs)) < 0
       || setDictItemSteal(dict, "depth_system_ts_us", PyLong_FromUnsignedLongLong(sample.depth.systemTsUs)) < 0
       || setDictItemSteal(dict, "sync_delta_us", PyLong_FromLongLong(sample.deltaSystemUs)) < 0) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

int PyOrbbecSdkReader_init(PyOrbbecSdkReader *self, PyObject *args, PyObject *kwargs) {
    const char *sdkRoot = "third_party/orbbecsdk/v1.10.27";
    const char *alignMode = "hardware";
    int width = 640;
    int height = 480;
    int fps = 30;
    int enableFrameSync = 0;
    double approxSyncMs = 50.0;
    int maxQueueSize = 10;
    int waitTimeoutMs = 1000;

    static const char *kwlist[] = {
        "sdk_root",
        "width",
        "height",
        "fps",
        "align_mode",
        "enable_frame_sync",
        "approx_sync_ms",
        "max_queue_size",
        "wait_timeout_ms",
        nullptr,
    };

    if(!PyArg_ParseTupleAndKeywords(
           args,
           kwargs,
           "|siiispdii",
           const_cast<char **>(kwlist),
           &sdkRoot,
           &width,
           &height,
           &fps,
           &alignMode,
           &enableFrameSync,
           &approxSyncMs,
           &maxQueueSize,
           &waitTimeoutMs
       )) {
        return -1;
    }

    ReaderConfig config;
    config.sdkRoot = sdkRoot;
    config.width = width;
    config.height = height;
    config.fps = fps;
    config.alignMode = alignMode;
    config.enableFrameSync = enableFrameSync != 0;
    config.approxSyncMs = approxSyncMs;
    config.maxQueueSize = maxQueueSize;
    config.waitTimeoutMs = waitTimeoutMs;

    try {
        self->reader = new OrbbecSdkReader(config);
    }
    catch(const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        self->reader = nullptr;
        return -1;
    }
    return 0;
}

void PyOrbbecSdkReader_dealloc(PyOrbbecSdkReader *self) {
    delete self->reader;
    self->reader = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *PyOrbbecSdkReader_next(PyOrbbecSdkReader *self, PyObject *) {
    if(self->reader == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "reader is not initialized");
        return nullptr;
    }
    try {
        return sampleToDict(self->reader->next());
    }
    catch(const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

PyObject *PyOrbbecSdkReader_get_intrinsic(PyOrbbecSdkReader *self, PyObject *) {
    if(self->reader == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "reader is not initialized");
        return nullptr;
    }
    return self->reader->intrinsicDict();
}

PyObject *PyOrbbecSdkReader_get_stats(PyOrbbecSdkReader *self, PyObject *) {
    if(self->reader == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "reader is not initialized");
        return nullptr;
    }
    return self->reader->statsDict();
}

PyObject *PyOrbbecSdkReader_close(PyOrbbecSdkReader *self, PyObject *) {
    if(self->reader != nullptr) {
        self->reader->close();
    }
    Py_RETURN_NONE;
}

PyMethodDef PyOrbbecSdkReader_methods[] = {
    {"next", reinterpret_cast<PyCFunction>(PyOrbbecSdkReader_next), METH_NOARGS, "Return next approximate-synced RGB/depth sample."},
    {"get_intrinsic", reinterpret_cast<PyCFunction>(PyOrbbecSdkReader_get_intrinsic), METH_NOARGS, "Return D2C camera intrinsics."},
    {"get_stats", reinterpret_cast<PyCFunction>(PyOrbbecSdkReader_get_stats), METH_NOARGS, "Return approximate-sync statistics."},
    {"close", reinterpret_cast<PyCFunction>(PyOrbbecSdkReader_close), METH_NOARGS, "Close the OrbbecSDK pipeline."},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject PyOrbbecSdkReaderType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

PyModuleDef moduleDef = {
    PyModuleDef_HEAD_INIT,
    "orbbecsdk_native",
    "Native OrbbecSDK v1 frame source wrapper.",
    -1,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_orbbecsdk_native(void) {
    PyOrbbecSdkReaderType.tp_name = "orbbecsdk_native.OrbbecSdkReader";
    PyOrbbecSdkReaderType.tp_basicsize = sizeof(PyOrbbecSdkReader);
    PyOrbbecSdkReaderType.tp_itemsize = 0;
    PyOrbbecSdkReaderType.tp_dealloc = reinterpret_cast<destructor>(PyOrbbecSdkReader_dealloc);
    PyOrbbecSdkReaderType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyOrbbecSdkReaderType.tp_doc = "OrbbecSDK v1 RGB-D frame reader";
    PyOrbbecSdkReaderType.tp_methods = PyOrbbecSdkReader_methods;
    PyOrbbecSdkReaderType.tp_init = reinterpret_cast<initproc>(PyOrbbecSdkReader_init);
    PyOrbbecSdkReaderType.tp_new = PyType_GenericNew;

    if(PyType_Ready(&PyOrbbecSdkReaderType) < 0) {
        return nullptr;
    }

    PyObject *module = PyModule_Create(&moduleDef);
    if(module == nullptr) {
        return nullptr;
    }

    if(_import_array() < 0) {
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(&PyOrbbecSdkReaderType);
    if(PyModule_AddObject(module, "OrbbecSdkReader", reinterpret_cast<PyObject *>(&PyOrbbecSdkReaderType)) < 0) {
        Py_DECREF(&PyOrbbecSdkReaderType);
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}
