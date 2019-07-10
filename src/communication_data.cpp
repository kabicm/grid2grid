namespace grid2grid {
// *********************
//     MESSAGE
// *********************
template <typename T>
message<T>::message(block<T> b, int rank)
    : b(b)
    , rank(rank) {}

template <typename T>
block<T> message<T>::get_block() const {
    return b;
}

template <typename T>
int message<T>::get_rank() const {
    return rank;
}

// implementing comparator
template <typename T>
bool message<T>::operator<(const message<T> &other) const {
    return get_rank() < other.get_rank() ||
           (get_rank() == other.get_rank() && b < other.get_block());
}

// ************************
//   COMMUNICATION DATA
// ************************
template <typename T>
communication_data<T>::communication_data(std::vector<message<T>> &&msgs,
                                          int n_ranks)
    : messages(std::forward<std::vector<message<T>>>(msgs))
    , n_ranks(n_ranks) {

    // std::cout << "constructor of communciation data invoked" << std::endl;
    dspls = std::vector<int>(n_ranks);
    counts = std::vector<int>(n_ranks);
    offset_per_message = std::vector<int>(messages.size());

    int offset = 0;

    for (unsigned i = 0; i < messages.size(); ++i) {
        const auto &m = messages[i];
        int rank = m.get_rank();
        block<T> b = m.get_block();
        offset_per_message[i] = offset;
        offset += b.total_size();
        // copy_block_to_buffer(b, buffer.begin() + offset);
        assert(b.non_empty());
        counts[rank] += b.total_size();
        total_size += b.total_size();

        // std::cout << "rank = " << rank << std::endl;
        // std::cout << "counts[rank] = " << counts[rank] << std::endl;
    }
    buffer = std::unique_ptr<T[]>(new T[total_size]);
    // buffer = std::vector<double, cosma::mpi_allocator<double>>(total_size);
    for (unsigned i = 1; i < (unsigned)n_ranks; ++i) {
        dspls[i] = dspls[i - 1] + counts[i - 1];
        // std::cout << "dpsls[rank] = " << dspls[i] << std::endl;
    }
    // std::cout << "total_size = " << total_size << std::endl;
}

template <typename T>
void copy_block_to_buffer(block<T> b, T *dest_ptr) {
    // std::cout << "copy block->buffer: " << b << std::endl;
    // std::cout << "copy block->buffer" << std::endl;
    memory::copy2D(b.size(), b.data, b.stride, dest_ptr, b.n_rows());
}

template <typename T>
void copy_block_from_buffer(T *src_ptr, block<T> &b) {
    // std::cout << "copy buffer->block" << std::endl;
    memory::copy2D(b.size(), src_ptr, b.n_rows(), b.data, b.stride);
}

template <typename T>
void communication_data<T>::copy_to_buffer() {
    // std::cout << "commuication data.copy_to_buffer()" << std::endl;
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < messages.size(); ++i) {
        const auto &m = messages[i];
        block<T> b = m.get_block();
        // int rank = m.get_rank();
        // std::cout << "rank = " << rank << std::endl;
        copy_block_to_buffer(b, data() + offset_per_message[i]);
    }
}

template <typename T>
void communication_data<T>::copy_from_buffer() {
    int offset = 0;
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < messages.size(); ++i) {
        const auto &m = messages[i];
        block<T> b = m.get_block();
        // int rank = m.get_rank();
        copy_block_from_buffer(data() + offset_per_message[i], b);
        offset += b.total_size();
    }
}

template <typename T>
T *communication_data<T>::data() {
    return buffer.get();
}
} // namespace grid2grid
