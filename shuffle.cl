/*
class LCGBijectiveFunction
{
public:
    LCGBijectiveFunction()
    {
    }

    template <class RandomGenerator>
    void init( uint64_t capacity, RandomGenerator& random_function )
    {
        modulus = roundUpPower2( capacity );
        // Must be odd so it is coprime to modulus
        multiplier = ( random_function() * 2 + 1 ) % modulus;
        addition = random_function() % modulus;
    }

    uint64_t getMappingRange() const
    {
        return modulus;
    }

    __host__ __device__ uint64_t operator()( uint64_t val ) const
    {
        // Modulus must be power of two
        assert( ( modulus & ( modulus - 1 ) ) == 0 );
        return ( ( val * multiplier ) + addition ) & ( modulus - 1 );
    }

    constexpr static bool isDeterministic()
    {
        return true;
    }

private:
    static uint64_t roundUpPower2( uint64_t a )
    {
        if( a & ( a - 1 ) )
        {
            uint64_t i;
            for( i = 0; a > 1; i++ )
            {
                a >>= 1ull;
            }
            return 1ull << ( i + 1ull );
        }
        return a;
    }

    uint64_t modulus;
    uint64_t multiplier;
    uint64_t addition;
};

template <class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
using LCGBijectiveScanShuffle =
    BijectiveFunctionScanShuffle<LCGBijectiveFunction, ContainerType, RandomGenerator>;



template <class BijectiveFunction, class ContainerType = thrust::device_vector<uint64_t>, class RandomGenerator = DefaultRandomGenerator>
class BijectiveFunctionScanShuffle : public Shuffle<ContainerType, RandomGenerator>
{
    constexpr static bool device =
        std::is_same<ContainerType, thrust::device_vector<typename ContainerType::value_type, typename ContainerType::allocator_type>>::value;
    BijectiveScanFuncs::cached_allocator<device> alloc;
    thrust::host_vector<BijectiveScanFuncs::KeyFlagTuple> tuple;

public:
    void shuffle( const ContainerType& in_container, ContainerType& out_container, uint64_t seed, uint64_t num ) override
    {
        using namespace BijectiveScanFuncs;
        assert( &in_container != &out_container );

        RandomGenerator random_function( seed );
        BijectiveFunction mapping_function;
        mapping_function.init( num, random_function );
        uint64_t capacity = mapping_function.getMappingRange();

        thrust::counting_iterator<uint64_t> indices( 0 );
        size_t m = num;

        WritePermutationFunctor<decltype( in_container.begin() ), decltype( out_container.begin() )> write_functor{
            m, in_container.begin(), out_container.begin()
        };
        auto output_it =
            thrust::make_transform_output_iterator( thrust::discard_iterator<uint64_t>(), write_functor );
        if constexpr( device )
        {
            thrust::transform_iterator<MakeTupleFunctor<BijectiveFunction>, decltype( indices ), KeyFlagTuple> tuple_it(
                indices, MakeTupleFunctor<BijectiveFunction>( m, mapping_function ) );
            thrust::inclusive_scan( thrust::cuda::par( alloc ), tuple_it, tuple_it + capacity,
                                    output_it, ScanOp() );
        }
        else
        {
            if( tuple.size() < capacity )
            {
                tuple.resize( capacity );
            }
            // Need to transform exactly once since computation is the bottleneck
            thrust::transform( thrust::tbb::par, indices, indices + capacity, tuple.begin(),
                               MakeTupleFunctor<BijectiveFunction>( m, mapping_function ) );
            // Explicitly call TBB scan to ensure parallel operation
            thrust::system::tbb::detail::inclusive_scan( thrust::tbb::par, tuple.begin(),
                                                         tuple.begin() + capacity, output_it, ScanOp() );
        }
    }

    bool supportsInPlace() const override
    {
        return false;
    }
};


*/

__kernel void shuffle_(
	__global float *arr
) {
	int i = get_global_id(0);
	arr[i] *= 2;
}
