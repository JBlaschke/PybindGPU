#ifndef PTR_WRAPPER_H
#define PTR_WRAPPER_H


template <class T>
class ptr_wrapper {
    public:
        ptr_wrapper()
        : ptr(nullptr), safe(false)
        {}

        ptr_wrapper(T * ptr, bool is_safe=false)
        : ptr(ptr), safe(is_safe)
        {}

        ptr_wrapper(const ptr_wrapper & other)
        : ptr(other.ptr), safe(other.is_safe())
        {}

        // Allocator
        void create(size_t N) {
            ptr = new T[N];
            safe = true;
        }

        // Pointer-like accessor functions
        T & operator* () const { return * ptr; }
        T * operator->() const { return   ptr; }
        T & operator[](std::size_t idx) const { return ptr[idx]; }

        // Accessor function
        T * get() const {
            return ptr;
        }

        // Conversion function 
        operator unsigned long() const {
            return (unsigned long) ptr;
        }

        // Deallocator
        void destroy() {
            delete ptr;
            safe = false;
        }

        // Return safety status of pointer
        bool is_safe() const { return safe; }

    private:
        T * ptr;
        bool safe;
};


template <class T>
struct obj_wrapper {
    T _obj;

    obj_wrapper(T & a_obj) : _obj(a_obj) {}
    obj_wrapper(T   a_obj) : _obj(a_obj) {}
    T & operator* () const { return _obj; }
    T & operator* ()       { return _obj; }
};

#endif