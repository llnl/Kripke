//
// Copyright (c) 2014-25, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the Kripke/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#ifndef KRIPKE_CORE_FIELD_H__
#define KRIPKE_CORE_FIELD_H__

#include <Kripke.h>
#include <Kripke/Core/VarLayout.h>
#include <Kripke/Core/DataStore.h>
#include <Kripke/Core/DomainVar.h>
#include <Kripke/Core/MemoryManager.h>
#include <Kripke/Core/Set.h>
#include <cstring>
#include <vector>

namespace Kripke {
namespace Core {
  template<typename ELEMENT, bool HOST_RESIDENT_NORMAL_GPU, typename ... IDX_TYPES>
  class FieldWithPolicy;
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
  template<typename ELEMENT, typename ... IDX_TYPES>
  class FieldWithDirectUmpireDeviceStorage;
#endif

  /**
   * Base class for Field which provides storage allocation
   */
  template<typename ELEMENT>
  class FieldStorage : public Kripke::Core::DomainVar {
    public:
      using ElementType = ELEMENT;

#if defined(KRIPKE_USE_CHAI)
      using ElementPtr = chai::ManagedArray<ELEMENT>;
#else
      using ElementPtr = ELEMENT*;
#endif

      using Layout1dType = RAJA::TypedLayout<RAJA::Index_type, camp::tuple<RAJA::Index_type>>;
      //FIXME: Remove the internal namespace when new RAJA release is out
      using View1dType = RAJA::internal::ViewBase<ElementType, ElementPtr, Layout1dType>;


      explicit FieldStorage(Kripke::Core::Set const &spanned_set
#if defined(KRIPKE_USE_UMPIRE)
          , Kripke::ExecutionSpace allocation_space = Kripke::CPU
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
          , bool direct_umpire_device_storage = false
#endif
#endif
          ) :
        m_set(&spanned_set)
#if defined(KRIPKE_USE_UMPIRE)
        , m_allocation_space(allocation_space)
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
        , m_direct_umpire_device_storage(direct_umpire_device_storage && allocation_space == Kripke::GPU)
#endif
#endif
      {

        // initialize our decomposition to match that of the specified set
        setup_initChunks(spanned_set);

        // allocate all of our chunks, and create layouts for each one
        size_t num_chunks = m_chunk_to_subdomain.size();
        m_chunk_to_size.resize(num_chunks, 0);
#if defined(KRIPKE_USE_CHAI) || defined(KRIPKE_USE_UMPIRE)
        m_chunk_to_data.resize(num_chunks);
#else
        m_chunk_to_data.resize(num_chunks, nullptr);
#endif

        for(size_t chunk_id = 0;chunk_id < num_chunks;++ chunk_id){

          // Get the size of the subdomain from the set
          SdomId sdom_id(m_chunk_to_subdomain[chunk_id]);
          size_t sdom_size = spanned_set.size(sdom_id);

          m_chunk_to_size[chunk_id] = sdom_size;
#if defined(KRIPKE_USE_UMPIRE)
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
          // Direct plane storage bypasses the CHAI GPU allocator (normally KRIPKE_DEVICE_POOL).
          // Used only for GPU-aware MPI i/j/k_plane buffers to avoid QuickPool non-base pointers.
          if(m_direct_umpire_device_storage){
#if defined(KRIPKE_USE_CHAI)
            m_chunk_to_data[chunk_id] = ElementPtr(
                sdom_size,
                {Kripke::CPU, Kripke::GPU}, // which CHAI execution spaces are being overridden
                {MemoryManager::getHostAllocator(), MemoryManager::getDirectDeviceAllocator()}, // matching Umpire allocators for the overridden spaces {HOST, NamedAllocationStrategy}
                Kripke::GPU); // initial allocation space
#else
            m_chunk_to_data[chunk_id].setHostAllocator(MemoryManager::getHostAllocator());
            m_chunk_to_data[chunk_id].setDeviceAllocator(MemoryManager::getDirectDeviceAllocator());
            m_chunk_to_data[chunk_id].allocate(sdom_size, Kripke::GPU);
#endif
          }
          else
#endif
          {
            m_chunk_to_data[chunk_id].allocate(sdom_size, m_allocation_space);
          }
#else
          m_chunk_to_data[chunk_id] = new ElementType[sdom_size];
#endif
        }
      }

      virtual ~FieldStorage(){
#if defined(KRIPKE_USE_CHAI)
// CHAI uses RAII semantics, hence no need to deallocate
#elif defined(KRIPKE_USE_UMPIRE)
        for(auto &chunk : m_chunk_to_data){
          chunk.deallocate();
        }
#else
        for(auto i : m_chunk_to_data){
          delete[] i;
        }
#endif
      }

      // Dissallow copy construction
      FieldStorage(FieldStorage<ElementType> const &) = delete;

      /**
       * Returns the number of elements in this subdomain.
       */
      RAJA_INLINE
      size_t size(Kripke::SdomId sdom_id) const {
        size_t chunk_id = m_subdomain_to_chunk[*sdom_id];
        return m_chunk_to_size[chunk_id];
      }


      RAJA_INLINE
      View1dType getView1d(Kripke::SdomId sdom_id) const {

        size_t chunk_id = m_subdomain_to_chunk[*sdom_id];

#if defined(KRIPKE_USE_CHAI)
        m_chunk_to_data[chunk_id].data(Kripke::CPU);
        ElementPtr ptr = m_chunk_to_data[chunk_id];
#elif defined(KRIPKE_USE_UMPIRE)
        ensureHostCurrent(chunk_id);
        ElementPtr ptr = m_chunk_to_data[chunk_id].getHostPtr();
#else
        ElementPtr ptr = m_chunk_to_data[chunk_id];
#endif
        size_t sdom_size = m_chunk_to_size[chunk_id];

        return View1dType(ptr, Layout1dType(sdom_size));
      }

      RAJA_INLINE
      ElementType *getHostData(Kripke::SdomId sdom_id) const {
        KRIPKE_ASSERT(*sdom_id < (int)m_subdomain_to_chunk.size(),
            "sdom_id(%d) >= num_subdomains(%d)",
            (int)*sdom_id,
            (int)(int)m_subdomain_to_chunk.size());
        size_t chunk_id = m_subdomain_to_chunk[*sdom_id];

#if defined(KRIPKE_USE_CHAI)
        return m_chunk_to_data[chunk_id].data(Kripke::CPU);
#elif defined(KRIPKE_USE_UMPIRE)
        ensureHostCurrent(chunk_id);
        return m_chunk_to_data[chunk_id].getHostPtr();
#else
        return m_chunk_to_data[chunk_id];
#endif
      }

      RAJA_INLINE
      ElementType const *getHostDataConst(Kripke::SdomId sdom_id) const {
        KRIPKE_ASSERT(*sdom_id < (int)m_subdomain_to_chunk.size(),
            "sdom_id(%d) >= num_subdomains(%d)",
            (int)*sdom_id,
            (int)(int)m_subdomain_to_chunk.size());
        size_t chunk_id = m_subdomain_to_chunk[*sdom_id];

#if defined(KRIPKE_USE_CHAI)
        return m_chunk_to_data[chunk_id].data(Kripke::CPU);
#elif defined(KRIPKE_USE_UMPIRE)
        ensureHostCurrent(chunk_id);
        return m_chunk_to_data[chunk_id].getHostPtr();
#else
        return m_chunk_to_data[chunk_id];
#endif
      }

      RAJA_INLINE
      ElementType *getDeviceData(Kripke::SdomId sdom_id) const {
        KRIPKE_ASSERT(*sdom_id < (int)m_subdomain_to_chunk.size(),
            "sdom_id(%d) >= num_subdomains(%d)",
            (int)*sdom_id,
            (int)(int)m_subdomain_to_chunk.size());
        size_t chunk_id = m_subdomain_to_chunk[*sdom_id];

#if defined(KRIPKE_USE_CHAI)
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
        if(m_direct_umpire_device_storage){
          return m_chunk_to_data[chunk_id].data(Kripke::GPU, false);
        }
#endif
#if defined(KRIPKE_USE_CUDA) || defined(KRIPKE_USE_HIP)
        return m_chunk_to_data[chunk_id].data(Kripke::GPU);
#else
        return m_chunk_to_data[chunk_id].data(Kripke::CPU);
#endif
#elif defined(KRIPKE_USE_UMPIRE) && (defined(KRIPKE_USE_CUDA) || defined(KRIPKE_USE_HIP))
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
        if(!m_direct_umpire_device_storage) {
          ensureDeviceCurrent(chunk_id);
        }
#else
        ensureDeviceCurrent(chunk_id);
#endif
        return m_chunk_to_data[chunk_id].getDevicePtr();
#else
        return m_chunk_to_data[chunk_id];
#endif
      }

      RAJA_INLINE
      ElementType *getData(Kripke::SdomId sdom_id) const {
        return getHostData(sdom_id);
      }

#if defined(KRIPKE_USE_UMPIRE)
      RAJA_INLINE
      Kripke::ExecutionSpace getAllocationSpace() const {
        return m_allocation_space;
      }

#ifdef KRIPKE_USE_GPU_AWARE_MPI
      RAJA_INLINE
      void registerDeviceTouch(Kripke::SdomId sdom_id) {
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
        if(m_direct_umpire_device_storage){
          return;
        }
#endif
        if(m_allocation_space == Kripke::GPU){
          KRIPKE_ASSERT(*sdom_id < (int)m_subdomain_to_chunk.size(),
              "sdom_id(%d) >= num_subdomains(%d)",
              (int)*sdom_id,
              (int)(int)m_subdomain_to_chunk.size());
          size_t chunk_id = m_subdomain_to_chunk[*sdom_id];
          m_chunk_to_data[chunk_id].registerTouch(Kripke::GPU);
        }
      }
#endif  // KRIPKE_USE_GPU_AWARE_MPI
#endif  // KRIPKE_USE_UMPIRE 


      RAJA_INLINE
      Kripke::Core::Set const &getSet() const {
        return *m_set;
      }

    protected:
#if defined(KRIPKE_USE_UMPIRE)
#if !defined(KRIPKE_USE_CHAI)
      class ChunkData {
        public:
          ChunkData() :
            m_host_allocator(MemoryManager::getHostAllocator()),
            m_device_allocator(MemoryManager::getDeviceAllocator())
          {}

          RAJA_INLINE
          void setHostAllocator(umpire::Allocator allocator) {
            m_host_allocator = allocator;
          }

          RAJA_INLINE
          void setDeviceAllocator(umpire::Allocator allocator) {
            m_device_allocator = allocator;
          }

          RAJA_INLINE
          void allocate(size_t num_elements, Kripke::ExecutionSpace allocation_space) {
            size_t bytes = num_elements*sizeof(ElementType);
            if(allocation_space == Kripke::GPU){
              m_device_ptr = allocateDeviceBuffer(bytes);
              m_last_space = Kripke::GPU;
            }
            else{
              m_host_ptr = allocateHostBuffer(bytes);
              m_last_space = Kripke::CPU;
            }
          }

          RAJA_INLINE
          void registerTouch(Kripke::ExecutionSpace space) const {
            m_last_space = space;
          }

          RAJA_INLINE
          void deallocate() {
            if(m_device_ptr != nullptr){
              m_device_allocator.deallocate(m_device_ptr);
              m_device_ptr = nullptr;
            }
            if(m_host_ptr != nullptr){
              m_host_allocator.deallocate(m_host_ptr);
              m_host_ptr = nullptr;
            }
          }

          RAJA_INLINE
          ElementType *getHostPtr() const {
            return m_host_ptr;
          }

          RAJA_INLINE
          void setHostPtr(ElementType *ptr) {
            m_host_ptr = ptr;
          }

          RAJA_INLINE
          ElementType *getDevicePtr() const {
            return m_device_ptr;
          }

          RAJA_INLINE
          void setDevicePtr(ElementType *ptr) {
            m_device_ptr = ptr;
          }

          RAJA_INLINE
          Kripke::ExecutionSpace getLastSpace() const {
            return m_last_space;
          }

          RAJA_INLINE
          ElementType *allocateHostBuffer(size_t bytes) {
            if(bytes == 0){
              return nullptr;
            }
            return static_cast<ElementType*>(m_host_allocator.allocate(bytes));
          }

          RAJA_INLINE
          ElementType *allocateDeviceBuffer(size_t bytes) {
            if(bytes == 0){
              return nullptr;
            }
            return static_cast<ElementType*>(m_device_allocator.allocate(bytes));
          }

        private:
          mutable ElementType *m_host_ptr = nullptr;
          mutable ElementType *m_device_ptr = nullptr;
          mutable Kripke::ExecutionSpace m_last_space = Kripke::CPU;
          umpire::Allocator m_host_allocator;
          umpire::Allocator m_device_allocator;
      };

      RAJA_INLINE
      void ensureHostCurrent(size_t chunk_id) const {
        auto &chunk = m_chunk_to_data[chunk_id];
        if(chunk.getDevicePtr() == nullptr ||
           chunk.getLastSpace() == Kripke::CPU){
          return;
        }

        if(chunk.getHostPtr() == nullptr){
          chunk.setHostPtr(chunk.allocateHostBuffer(m_chunk_to_size[chunk_id]*sizeof(ElementType)));
        }
        synchronizeDevice();
        MemoryManager::copy(chunk.getHostPtr(),
                            chunk.getDevicePtr(),
                            m_chunk_to_size[chunk_id]*sizeof(ElementType));
        chunk.registerTouch(Kripke::CPU);
      }

      RAJA_INLINE
      void ensureDeviceCurrent(size_t chunk_id) const {
        auto &chunk = m_chunk_to_data[chunk_id];
        if(chunk.getLastSpace() == Kripke::GPU){
          return;
        }

        if(chunk.getDevicePtr() == nullptr){
          chunk.setDevicePtr(chunk.allocateDeviceBuffer(m_chunk_to_size[chunk_id]*sizeof(ElementType)));
        }
        MemoryManager::copy(chunk.getDevicePtr(),
                            chunk.getHostPtr(),
                            m_chunk_to_size[chunk_id]*sizeof(ElementType));
        chunk.registerTouch(Kripke::GPU);
      }

      RAJA_INLINE
      static void synchronizeDevice() {
#if defined(KRIPKE_USE_CUDA)
        RAJA::synchronize<RAJA::cuda_synchronize>();
#elif defined(KRIPKE_USE_HIP)
        RAJA::synchronize<RAJA::hip_synchronize>();
#endif
      }
#endif  // !KRIPKE_USE_CHAI
#endif  // KRIPKE_USE_UMPIRE

      Kripke::Core::Set const *m_set;
      std::vector<size_t> m_chunk_to_size;
#if defined(KRIPKE_USE_CHAI)
      std::vector<ElementPtr> m_chunk_to_data;
      Kripke::ExecutionSpace m_allocation_space;
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
      bool m_direct_umpire_device_storage;
#endif
#elif defined(KRIPKE_USE_UMPIRE)
      mutable std::vector<ChunkData> m_chunk_to_data;
      Kripke::ExecutionSpace m_allocation_space;
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
      bool m_direct_umpire_device_storage;
#endif
#else
      std::vector<ElementPtr> m_chunk_to_data;
#endif
	  };

  /**
   * Defines a multi-dimensional data field defined over a Set
   */
  template<typename ELEMENT, typename ... IDX_TYPES>
  class Field : public Kripke::Core::FieldStorage<ELEMENT> {
    public:

      using Parent = Kripke::Core::FieldStorage<ELEMENT>;

      using ElementType = ELEMENT;
      static constexpr bool host_resident_normal_gpu = false;

#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
      static constexpr bool direct_umpire_device_storage = false;
#endif

      using ElementPtr = typename Parent::ElementPtr;

      static constexpr size_t NumDims = sizeof...(IDX_TYPES);

      using DefaultLayoutType = RAJA::TypedLayout<RAJA::Index_type, camp::tuple<IDX_TYPES...>>;

      //FIXME: Remove the internal namespace when new RAJA release is out
      using DefaultViewType = RAJA::internal::ViewBase<ElementType, ElementPtr, DefaultLayoutType>;
      using DeviceViewType = RAJA::internal::ViewBase<ElementType, ElementType *, DefaultLayoutType>;

      template<typename Order>
      Field(Kripke::Core::Set const &spanned_set, Order) :
        Parent(spanned_set)
      {
        setupLayouts<Order>(spanned_set);
      }

#if defined(KRIPKE_USE_UMPIRE)
      template<typename Order>
      Field(Kripke::Core::Set const &spanned_set,
            Kripke::ExecutionSpace allocation_space,
            Order) :
        Parent(spanned_set, allocation_space)
      {
        setupLayouts<Order>(spanned_set);
      }

#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
      template<typename Order>
      Field(Kripke::Core::Set const &spanned_set,
            Kripke::ExecutionSpace allocation_space,
            bool direct_umpire_device_storage_arg,
            Order) :
        Parent(spanned_set, allocation_space, direct_umpire_device_storage_arg)
      {
        setupLayouts<Order>(spanned_set);
      }
#endif
#endif  // KRIPKE_USE_UMPIRE

      template<typename Order>
      void setupLayouts(Kripke::Core::Set const &spanned_set) {
        KRIPKE_ASSERT(NumDims == spanned_set.getNumDimensions(),
            "Number of dimensions must match between Field<%d> and Set<%d>\n",
            (int)NumDims, (int)spanned_set.getNumDimensions());

        auto perm = LayoutInfo<Order, IDX_TYPES...>::getPermutation();

        // create layouts for each chunk
        size_t num_chunks = Parent::m_chunk_to_subdomain.size();
        m_chunk_to_layout.resize(num_chunks);
        for(size_t chunk_id = 0;chunk_id < num_chunks;++ chunk_id){

          // Create a layout using dim sizes from the Set, and permutation
          // defined by the layout function
          SdomId sdom_id(Parent::m_chunk_to_subdomain[chunk_id]);
          std::array<RAJA::Index_type, NumDims> sizes;
          for(size_t dim = 0;dim < NumDims;++ dim){
            sizes[dim] = spanned_set.dimSize(sdom_id, dim);
          }

          RAJA::Layout<NumDims, RAJA::Index_type> &layout =
              m_chunk_to_layout[chunk_id];
          layout = RAJA::make_permuted_layout<NumDims,RAJA::Index_type>(sizes, perm);
        }
      }

      virtual ~Field(){

      }



      RAJA_INLINE
      DefaultViewType getView(Kripke::SdomId sdom_id) const {

        size_t chunk_id = Parent::m_subdomain_to_chunk[*sdom_id];

#if defined(KRIPKE_USE_CHAI)
        Parent::m_chunk_to_data[chunk_id].data(Kripke::CPU);
        auto ptr = Parent::m_chunk_to_data[chunk_id];
#elif defined(KRIPKE_USE_UMPIRE)
        Parent::ensureHostCurrent(chunk_id);
        auto ptr = Parent::m_chunk_to_data[chunk_id].getHostPtr();
#else
        auto ptr = Parent::m_chunk_to_data[chunk_id];
#endif
        auto layout = m_chunk_to_layout[chunk_id];

        return DefaultViewType(ptr, layout);
      }


      RAJA_INLINE
      DeviceViewType getDeviceView(Kripke::SdomId sdom_id) const {

        size_t chunk_id = Parent::m_subdomain_to_chunk[*sdom_id];
        auto layout = m_chunk_to_layout[chunk_id];

        auto ptr = Parent::getDeviceData(sdom_id);

        return DeviceViewType(ptr, layout);
      }


      template<typename Order>
      RAJA_INLINE
      auto getViewOrder(Kripke::SdomId sdom_id) const ->
        ViewType<Order, ElementType, ElementType *, IDX_TYPES...>
      {
        size_t chunk_id = Parent::m_subdomain_to_chunk[*sdom_id];

        using LInfo = LayoutInfo<Order, IDX_TYPES...>;
        using LType = typename LInfo::Layout;
        using OrderedViewType = ViewType<Order, ElementType, ElementType *, IDX_TYPES...>;

        LType layout = RAJA::make_stride_one<LInfo::stride_one_dim>(m_chunk_to_layout[chunk_id]);

#if (defined(KRIPKE_USE_HIP) || defined(KRIPKE_USE_CUDA)) && defined(KRIPKE_USE_CHAI)
        if(Parent::m_allocation_space == Kripke::GPU){
          return OrderedViewType(Parent::getDeviceData(sdom_id), layout);
        }
        return OrderedViewType(Parent::m_chunk_to_data[chunk_id].data(Kripke::CPU), layout);
#elif (defined(KRIPKE_USE_HIP) || defined(KRIPKE_USE_CUDA)) && defined(KRIPKE_USE_UMPIRE)
        if(Parent::m_allocation_space == Kripke::GPU){
          return OrderedViewType(Parent::getDeviceData(sdom_id), layout);
        }
        return OrderedViewType(Parent::getHostData(sdom_id), layout);
#else
        return OrderedViewType(Parent::getHostData(sdom_id), layout);
#endif
      }



      RAJA_INLINE
      void dump() const {
        printf("Field<>:\n");
        printf("  name:  %s\n", BaseVar::getName().c_str());
        printf("  m_set: %p\n", Parent::m_set);

        printf("  m_chunk_to_size: ");
        for(auto x : Parent::m_chunk_to_size){printf("%lu ", (unsigned long)x);}
        printf("\n");

#if defined(KRIPKE_USE_UMPIRE) && !defined(KRIPKE_USE_CHAI)
        printf("  m_chunk_to_data(host): ");
        for(auto const &x : Parent::m_chunk_to_data){printf("%p ", (void*)x.getHostPtr());}
        printf("\n");
        printf("  m_chunk_to_data(device): ");
        for(auto const &x : Parent::m_chunk_to_data){printf("%p ", (void*)x.getDevicePtr());}
        printf("\n");
#else
        printf("  m_chunk_to_data: ");
        for(auto x : Parent::m_chunk_to_data){printf("%p ", (void*)x);}
        printf("\n");
#endif

        for(size_t chunk_id = 0;chunk_id < Parent::m_chunk_to_data.size();++ chunk_id){

          SdomId sdom_id(DomainVar::m_chunk_to_subdomain[chunk_id]);

          ElementType *ptr = Parent::getHostData(sdom_id);

          printf("Chunk %d Data: ", (int)chunk_id);
          for(size_t i = 0;i < Parent::m_chunk_to_size[chunk_id];++ i){
            printf(" %e", ptr[i]);
          }
          printf("\n");
        }

        Kripke::Core::DomainVar::dump();
      }

    protected:
      std::vector<DefaultLayoutType> m_chunk_to_layout;
  };

  template<typename ELEMENT, bool HOST_RESIDENT_NORMAL_GPU, typename ... IDX_TYPES>
  class FieldWithPolicy : public Kripke::Core::Field<ELEMENT, IDX_TYPES...> {
    public:
      using Parent = Kripke::Core::Field<ELEMENT, IDX_TYPES...>;
      using Parent::Parent;
      static constexpr bool host_resident_normal_gpu =
          HOST_RESIDENT_NORMAL_GPU;
#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
      static constexpr bool direct_umpire_device_storage = false;
#endif
  };

#ifdef KRIPKE_USE_DIRECT_UMPIRE_PLANE_STORAGE
  template<typename ELEMENT, typename ... IDX_TYPES>
  class FieldWithDirectUmpireDeviceStorage : public Kripke::Core::Field<ELEMENT, IDX_TYPES...> {
    public:
      using Parent = Kripke::Core::Field<ELEMENT, IDX_TYPES...>;
      using Parent::Parent;
      static constexpr bool host_resident_normal_gpu = false;
      static constexpr bool direct_umpire_device_storage = true;
  };
#endif

} } // namespace

#endif
