#ifndef HOST_WRAPPER_H
#define HOST_WRAPPER_H

template<class T, class E>
class Allocator {
    public:
        Allocator() : _ptr(NULL) {}
        virtual E allocate(size_t elts) {}
        virtual E deallocate() {}
        T * ptr() {return _ptr;}

    protected:
        T * _ptr;
        size_t _bytes;
};


template<class T>
class PagelockedAllocator : public Allocator<T, cudaError_t> {
    public:
        cudaError_t allocate(size_t elts) override {

        }

        cudaError_t deallocate() override {

        }
}


// template<class T>
// class HostArray {
//     public:
// 
//         HostArray() = delete;
//         HostArray(const HostArray & ) = delete;
// 
//         HostArray(std::vector<ssize_t> & shape)
//         : m_shape{shape}, device_allocated(false) {
//             // total size
//             m_size = std::accumulate(
//                 shape.begin(), shape.end(), 1,
//                 std::multiplies<ssize_t>()
//             );
//             // define array strides, assuming c-order
//             m_ndim = shape.size();
//             m_strides = std::vector<ssize_t>(m_ndim);
//             ssize_t stride = sizeof(T);
//             for (int i = m_ndim - 1; i >= 0; i--) {
//                 m_strides[i] = stride;
//                 stride = stride * shape[i];
//             }
//             // allocate data
//             host_ptr = new T[m_size];
//             // allocation status
//             host_allocated = true;
//             device_allocated = false;
//         };
// 
// 
//         T * host_data() { return host_ptr; }
//         ssize_t size() const { return m_size; }
//         const std::vector<ssize_t> & shape() const { return m_shape; };
//         const std::vector<ssize_t> & strides() const { return m_strides; };
// 
//         py::buffer_info buffer_info() {
//             return py::buffer_info(
//                 /* Pointer to buffer */
//                 host_ptr,
//                 /* Size of one scalar */
//                 sizeof(T),
//                 /* Python struct-style format descriptor */
//                 py::format_descriptor<T>::format(),
//                 /* Number of dimensions */
//                 m_ndim,
//                 /* Buffer dimensions */
//                 m_shape,
//                 /* Strides (in bytes) for each index */
//                 m_strides
//             );
//         }
// 
//     private:
//         ssize_t m_size;
//         std::vector<ssize_t> m_shape;
//         int m_ndim;
//         std::vector<ssize_t> m_strides;
// 
//         T * host_ptr;
// };

#endif