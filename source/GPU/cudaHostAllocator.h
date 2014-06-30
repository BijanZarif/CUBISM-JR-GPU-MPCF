/* *
 * cudaHostAllocator.h
 *
 * Created by Fabian Wermelinger on 06/06/14.
 * Copyright 2014 ETH Zurich. All rights reserved.
 * */
#include <cstdlib>
#include <memory>
#include <limits>


extern "C"
{
    void *_cudaAllocHost(const std::size_t bytes);
    void  _cudaFreeHost(void *ptr);
}


template <typename T>
class cudaHostAllocator
{
    public :
        //    typedefs
        typedef T value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

    public :
        //    convert a cudaHostAllocator<T> to cudaHostAllocator<U>
        template<typename U>
            struct rebind {
                typedef cudaHostAllocator<U> other;
            };

    public :
        inline explicit cudaHostAllocator() {}
        inline ~cudaHostAllocator() {}
        inline explicit cudaHostAllocator(cudaHostAllocator const&) {}
        template<typename U>
            inline explicit cudaHostAllocator(cudaHostAllocator<U> const&) {}

        //    address
        inline pointer address(reference r) { return &r; }
        inline const_pointer address(const_reference r) { return &r; }

        //    memory allocation
        inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0)
        {
            return reinterpret_cast<pointer>(_cudaAllocHost(cnt * sizeof(T)));
        }
        inline void deallocate(pointer p, size_type)
        {
            _cudaFreeHost(p);
        }

        //    size
        inline size_type max_size() const {
            return std::numeric_limits<size_type>::max() / sizeof(T);
        }

        //    construction/destruction
        inline void construct(pointer p, const T& t) { new(p) T(t); }
        inline void destroy(pointer p) { p->~T(); }

        inline bool operator==(cudaHostAllocator const&) { return true; }
        inline bool operator!=(cudaHostAllocator const& a) { return !operator==(a); }
};
