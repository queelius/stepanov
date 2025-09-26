#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <variant>
#include <vector>
#include "concepts.hpp"
#include "error.hpp"

namespace stepanov {

// Serialization concepts
template<typename T>
concept serializable = requires(T t, std::ostream& os) {
    { t.serialize(os) } -> std::same_as<void>;
};

template<typename T>
concept deserializable = requires(T t, std::istream& is) {
    { T::deserialize(is) } -> std::same_as<T>;
};

template<typename T>
concept binary_serializable = std::is_trivially_copyable_v<T>;

// Binary format serialization
class binary_writer {
private:
    std::vector<uint8_t> buffer;
    std::size_t position;

    void ensure_capacity(std::size_t additional) {
        if (position + additional > buffer.size()) {
            buffer.resize(std::max(buffer.size() * 2, position + additional));
        }
    }

public:
    binary_writer() : position(0) {
        buffer.reserve(1024);
    }

    // Write primitive types
    template<typename T>
        requires binary_serializable<T>
    void write(const T& value) {
        ensure_capacity(sizeof(T));
        std::memcpy(buffer.data() + position, &value, sizeof(T));
        position += sizeof(T);
    }

    // Write with endianness conversion (manual byte swap for C++20)
    template<std::integral T>
    void write_be(T value) {
        if constexpr (std::endian::native == std::endian::little) {
            // Manual byte swap for portability
            auto bytes = reinterpret_cast<uint8_t*>(&value);
            for (std::size_t i = 0; i < sizeof(T) / 2; ++i) {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }
        write(value);
    }

    template<std::integral T>
    void write_le(T value) {
        if constexpr (std::endian::native == std::endian::big) {
            // Manual byte swap for portability
            auto bytes = reinterpret_cast<uint8_t*>(&value);
            for (std::size_t i = 0; i < sizeof(T) / 2; ++i) {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }
        write(value);
    }

    // Write strings
    void write_string(std::string_view str) {
        write<uint32_t>(static_cast<uint32_t>(str.size()));
        ensure_capacity(str.size());
        std::memcpy(buffer.data() + position, str.data(), str.size());
        position += str.size();
    }

    // Write arrays
    template<typename T, std::size_t N>
        requires binary_serializable<T>
    void write_array(const std::array<T, N>& arr) {
        ensure_capacity(sizeof(T) * N);
        std::memcpy(buffer.data() + position, arr.data(), sizeof(T) * N);
        position += sizeof(T) * N;
    }

    // Write vectors
    template<typename T>
        requires binary_serializable<T>
    void write_vector(const std::vector<T>& vec) {
        write<uint32_t>(static_cast<uint32_t>(vec.size()));
        ensure_capacity(sizeof(T) * vec.size());
        std::memcpy(buffer.data() + position, vec.data(), sizeof(T) * vec.size());
        position += sizeof(T) * vec.size();
    }

    // Write raw bytes
    void write_bytes(const uint8_t* data, std::size_t size) {
        ensure_capacity(size);
        std::memcpy(buffer.data() + position, data, size);
        position += size;
    }

    // Variable-length integer encoding (like protobuf)
    void write_varint(uint64_t value) {
        while (value >= 0x80) {
            write<uint8_t>(static_cast<uint8_t>(value | 0x80));
            value >>= 7;
        }
        write<uint8_t>(static_cast<uint8_t>(value));
    }

    // Get the buffer
    std::vector<uint8_t> get_buffer() const {
        std::vector<uint8_t> result(buffer.begin(), buffer.begin() + position);
        return result;
    }

    std::size_t size() const { return position; }
    void clear() { position = 0; }
};

class binary_reader {
private:
    const uint8_t* data;
    std::size_t size;
    std::size_t position;

public:
    binary_reader(const uint8_t* d, std::size_t s)
        : data(d), size(s), position(0) {}

    explicit binary_reader(const std::vector<uint8_t>& buffer)
        : data(buffer.data()), size(buffer.size()), position(0) {}

    // Read primitive types
    template<typename T>
        requires binary_serializable<T>
    expected<T, std::string> read() {
        if (position + sizeof(T) > size) {
            return expected<T, std::string>::failure("Buffer underflow");
        }

        T value;
        std::memcpy(&value, data + position, sizeof(T));
        position += sizeof(T);
        return value;
    }

    // Read with endianness conversion
    template<std::integral T>
    expected<T, std::string> read_be() {
        auto result = read<T>();
        if (result.has_value()) {
            if constexpr (std::endian::native == std::endian::little) {
                T value = result.value();
                auto bytes = reinterpret_cast<uint8_t*>(&value);
                for (std::size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
                return value;
            }
        }
        return result;
    }

    template<std::integral T>
    expected<T, std::string> read_le() {
        auto result = read<T>();
        if (result.has_value()) {
            if constexpr (std::endian::native == std::endian::big) {
                T value = result.value();
                auto bytes = reinterpret_cast<uint8_t*>(&value);
                for (std::size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
                return value;
            }
        }
        return result;
    }

    // Read strings
    std::optional<std::string> read_string() {
        auto size_result = read<uint32_t>();
        if (!size_result.has_value()) {
            return std::nullopt;
        }

        uint32_t str_size = size_result.value();
        if (position + str_size > size) {
            return std::nullopt;
        }

        std::string result(reinterpret_cast<const char*>(data + position), str_size);
        position += str_size;
        return result;
    }

    // Read arrays
    template<typename T, std::size_t N>
        requires binary_serializable<T>
    expected<std::array<T, N>, std::string> read_array() {
        if (position + sizeof(T) * N > size) {
            return expected<std::array<T, N>, std::string>::failure("Buffer underflow");
        }

        std::array<T, N> result;
        std::memcpy(result.data(), data + position, sizeof(T) * N);
        position += sizeof(T) * N;
        return result;
    }

    // Read vectors
    template<typename T>
        requires binary_serializable<T>
    expected<std::vector<T>, std::string> read_vector() {
        auto size_result = read<uint32_t>();
        if (!size_result.has_value()) {
            return expected<std::vector<T>, std::string>::failure("Failed to read vector size");
        }

        uint32_t vec_size = size_result.value();
        if (position + sizeof(T) * vec_size > size) {
            return expected<std::vector<T>, std::string>::failure("Buffer underflow reading vector");
        }

        std::vector<T> result(vec_size);
        std::memcpy(result.data(), data + position, sizeof(T) * vec_size);
        position += sizeof(T) * vec_size;
        return result;
    }

    // Read raw bytes
    expected<std::vector<uint8_t>, std::string> read_bytes(std::size_t count) {
        if (position + count > size) {
            return expected<std::vector<uint8_t>, std::string>::failure("Buffer underflow");
        }

        std::vector<uint8_t> result(data + position, data + position + count);
        position += count;
        return result;
    }

    // Variable-length integer decoding
    expected<uint64_t, std::string> read_varint() {
        uint64_t result = 0;
        int shift = 0;

        while (position < size) {
            uint8_t byte = data[position++];
            result |= static_cast<uint64_t>(byte & 0x7F) << shift;

            if ((byte & 0x80) == 0) {
                return result;
            }

            shift += 7;
            if (shift >= 64) {
                return expected<uint64_t, std::string>::failure("Varint too large");
            }
        }

        return expected<uint64_t, std::string>::failure("Unexpected end of buffer");
    }

    bool has_remaining() const { return position < size; }
    std::size_t remaining() const { return size - position; }
    std::size_t tell() const { return position; }
    void seek(std::size_t pos) { position = std::min(pos, size); }
};

// JSON serialization
class json_value {
public:
    using null_t = std::monostate;
    using object_t = std::map<std::string, json_value>;
    using array_t = std::vector<json_value>;

private:
    std::variant<null_t, bool, int64_t, double, std::string, object_t, array_t> data;

public:
    // Constructors
    json_value() : data(null_t{}) {}
    json_value(std::nullptr_t) : data(null_t{}) {}
    json_value(bool b) : data(b) {}
    json_value(int v) : data(static_cast<int64_t>(v)) {}
    json_value(int64_t v) : data(v) {}
    json_value(double d) : data(d) {}
    json_value(const char* s) : data(std::string(s)) {}
    json_value(std::string s) : data(std::move(s)) {}
    json_value(object_t obj) : data(std::move(obj)) {}
    json_value(array_t arr) : data(std::move(arr)) {}

    // Type checking
    bool is_null() const { return std::holds_alternative<null_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_number() const {
        return std::holds_alternative<int64_t>(data) ||
               std::holds_alternative<double>(data);
    }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_object() const { return std::holds_alternative<object_t>(data); }
    bool is_array() const { return std::holds_alternative<array_t>(data); }

    // Accessors
    bool as_bool() const { return std::get<bool>(data); }
    int64_t as_int() const { return std::get<int64_t>(data); }
    double as_double() const {
        if (std::holds_alternative<int64_t>(data)) {
            return static_cast<double>(std::get<int64_t>(data));
        }
        return std::get<double>(data);
    }
    const std::string& as_string() const { return std::get<std::string>(data); }
    const object_t& as_object() const { return std::get<object_t>(data); }
    const array_t& as_array() const { return std::get<array_t>(data); }

    // Object access
    json_value& operator[](const std::string& key) {
        if (!is_object()) {
            data = object_t{};
        }
        return std::get<object_t>(data)[key];
    }

    const json_value& operator[](const std::string& key) const {
        static json_value null_value;
        if (!is_object()) return null_value;

        const auto& obj = std::get<object_t>(data);
        auto it = obj.find(key);
        return it != obj.end() ? it->second : null_value;
    }

    // Array access
    json_value& operator[](std::size_t index) {
        if (!is_array()) {
            data = array_t{};
        }
        auto& arr = std::get<array_t>(data);
        if (index >= arr.size()) {
            arr.resize(index + 1);
        }
        return arr[index];
    }

    const json_value& operator[](std::size_t index) const {
        static json_value null_value;
        if (!is_array()) return null_value;

        const auto& arr = std::get<array_t>(data);
        return index < arr.size() ? arr[index] : null_value;
    }

    // Stringify
    std::string to_string(int indent = 0) const {
        std::ostringstream ss;
        write_to(ss, indent, 0);
        return ss.str();
    }

private:
    void write_to(std::ostream& os, int indent, int current_indent) const {
        std::string indent_str(current_indent, ' ');

        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;

            if constexpr (std::is_same_v<T, null_t>) {
                os << "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                os << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, int64_t>) {
                os << v;
            } else if constexpr (std::is_same_v<T, double>) {
                os << v;
            } else if constexpr (std::is_same_v<T, std::string>) {
                os << '"' << escape_string(v) << '"';
            } else if constexpr (std::is_same_v<T, object_t>) {
                os << '{';
                if (indent > 0) os << '\n';

                bool first = true;
                for (const auto& [key, value] : v) {
                    if (!first) {
                        os << ',';
                        if (indent > 0) os << '\n';
                    }
                    first = false;

                    if (indent > 0) os << std::string(current_indent + indent, ' ');
                    os << '"' << escape_string(key) << "\":";
                    if (indent > 0) os << ' ';
                    value.write_to(os, indent, current_indent + indent);
                }

                if (indent > 0 && !v.empty()) {
                    os << '\n' << indent_str;
                }
                os << '}';
            } else if constexpr (std::is_same_v<T, array_t>) {
                os << '[';
                if (indent > 0 && !v.empty()) os << '\n';

                bool first = true;
                for (const auto& elem : v) {
                    if (!first) {
                        os << ',';
                        if (indent > 0) os << '\n';
                    }
                    first = false;

                    if (indent > 0) os << std::string(current_indent + indent, ' ');
                    elem.write_to(os, indent, current_indent + indent);
                }

                if (indent > 0 && !v.empty()) {
                    os << '\n' << indent_str;
                }
                os << ']';
            }
        }, data);
    }

    static std::string escape_string(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\b': result += "\\b"; break;
                case '\f': result += "\\f"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:
                    if (c >= 0x20 && c <= 0x7E) {
                        result += c;
                    } else {
                        result += "\\u00";
                        result += "0123456789ABCDEF"[(c >> 4) & 0xF];
                        result += "0123456789ABCDEF"[c & 0xF];
                    }
            }
        }
        return result;
    }
};

// Type registry for polymorphic serialization
class type_registry {
private:
    using creator_func = std::function<std::unique_ptr<void>()>;
    using serializer_func = std::function<void(const void*, binary_writer&)>;
    using deserializer_func = std::function<std::unique_ptr<void>(binary_reader&)>;

    struct type_info {
        std::string name;
        creator_func creator;
        serializer_func serializer;
        deserializer_func deserializer;
    };

    std::unordered_map<std::type_index, type_info> types;
    std::unordered_map<std::string, std::type_index> name_to_type;

public:
    template<typename T>
    void register_type(const std::string& name) {
        auto type_idx = std::type_index(typeid(T));

        type_info info;
        info.name = name;

        info.creator = []() -> std::unique_ptr<void> {
            return std::make_unique<T>();
        };

        info.serializer = [](const void* obj, binary_writer& writer) {
            const T* typed_obj = static_cast<const T*>(obj);
            // Assume T has a serialize method or is binary_serializable
            if constexpr (binary_serializable<T>) {
                writer.write(*typed_obj);
            }
        };

        info.deserializer = [](binary_reader& reader) -> std::unique_ptr<void> {
            if constexpr (binary_serializable<T>) {
                auto result = reader.read<T>();
                if (result.has_value()) {
                    return std::make_unique<T>(result.value());
                }
            }
            return nullptr;
        };

        types[type_idx] = info;
        name_to_type[name] = type_idx;
    }

    template<typename T>
    void serialize(const T& obj, binary_writer& writer) {
        auto type_idx = std::type_index(typeid(T));
        auto it = types.find(type_idx);

        if (it != types.end()) {
            writer.write_string(it->second.name);
            it->second.serializer(&obj, writer);
        }
    }

    template<typename BaseType>
    std::unique_ptr<BaseType> deserialize(binary_reader& reader) {
        auto name_result = reader.read_string();
        if (!name_result.has_value()) {
            return nullptr;
        }

        auto name_it = name_to_type.find(name_result.value());
        if (name_it == name_to_type.end()) {
            return nullptr;
        }

        auto type_it = types.find(name_it->second);
        if (type_it == types.end()) {
            return nullptr;
        }

        auto result = type_it->second.deserializer(reader);
        return std::unique_ptr<BaseType>(static_cast<BaseType*>(result.release()));
    }
};

// Version-aware serialization
class versioned_serializer {
private:
    static constexpr uint32_t magic_number = 0x53545056;  // "STPV"
    uint32_t version;

public:
    explicit versioned_serializer(uint32_t ver = 1) : version(ver) {}

    template<typename T>
    std::vector<uint8_t> serialize(const T& obj) {
        binary_writer writer;

        // Write header
        writer.write(magic_number);
        writer.write(version);

        // Write object
        if constexpr (binary_serializable<T>) {
            writer.write(obj);
        } else if constexpr (serializable<T>) {
            std::ostringstream ss;
            obj.serialize(ss);
            writer.write_string(ss.str());
        }

        return writer.get_buffer();
    }

    template<typename T>
    expected<T, std::string> deserialize(const std::vector<uint8_t>& data) {
        binary_reader reader(data);

        // Read and verify header
        auto magic_result = reader.read<uint32_t>();
        if (!magic_result.has_value() || magic_result.value() != magic_number) {
            return expected<T, std::string>::failure("Invalid magic number");
        }

        auto version_result = reader.read<uint32_t>();
        if (!version_result.has_value()) {
            return expected<T, std::string>::failure("Failed to read version");
        }

        // Handle version compatibility
        if (version_result.value() > version) {
            return expected<T, std::string>::failure("Unsupported version");
        }

        // Read object
        if constexpr (binary_serializable<T>) {
            return reader.read<T>();
        } else if constexpr (deserializable<T>) {
            auto str_result = reader.read_string();
            if (str_result.has_value()) {
                std::istringstream ss(str_result.value());
                return T::deserialize(ss);
            }
        }

        return expected<T, std::string>::failure("Deserialization failed");
    }
};

// Zero-copy deserialization view
template<typename T>
    requires binary_serializable<T>
class zero_copy_view {
private:
    const uint8_t* data;
    std::size_t count;

public:
    zero_copy_view(const uint8_t* d, std::size_t n)
        : data(d), count(n / sizeof(T)) {}

    const T& operator[](std::size_t index) const {
        return *reinterpret_cast<const T*>(data + index * sizeof(T));
    }

    std::size_t size() const { return count; }

    const T* begin() const { return reinterpret_cast<const T*>(data); }
    const T* end() const { return begin() + count; }
};

// Compression wrapper (simplified RLE for demonstration)
class compressed_serializer {
public:
    template<typename T>
        requires std::integral<T>
    static std::vector<uint8_t> compress_rle(const std::vector<T>& data) {
        binary_writer writer;

        if (data.empty()) {
            writer.write<uint32_t>(0);
            return writer.get_buffer();
        }

        writer.write<uint32_t>(static_cast<uint32_t>(data.size()));

        std::size_t i = 0;
        while (i < data.size()) {
            T value = data[i];
            std::size_t count = 1;

            while (i + count < data.size() && data[i + count] == value) {
                ++count;
            }

            writer.write_varint(count);
            writer.write(value);

            i += count;
        }

        return writer.get_buffer();
    }

    template<typename T>
        requires std::integral<T>
    static expected<std::vector<T>, std::string> decompress_rle(const std::vector<uint8_t>& compressed) {
        binary_reader reader(compressed);

        auto size_result = reader.read<uint32_t>();
        if (!size_result.has_value()) {
            return expected<std::vector<T>, std::string>::failure("Failed to read size");
        }

        std::vector<T> result;
        result.reserve(size_result.value());

        while (reader.has_remaining() && result.size() < size_result.value()) {
            auto count_result = reader.read_varint();
            if (!count_result.has_value()) {
                return expected<std::vector<T>, std::string>::failure("Failed to read count");
            }

            auto value_result = reader.read<T>();
            if (!value_result.has_value()) {
                return expected<std::vector<T>, std::string>::failure("Failed to read value");
            }

            for (std::size_t i = 0; i < count_result.value(); ++i) {
                result.push_back(value_result.value());
            }
        }

        return result;
    }
};

} // namespace stepanov