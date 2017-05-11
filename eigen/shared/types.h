/*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* [ MIT license: http://www.opensource.org/licenses/mit-license.php ]
*/

#pragma once

#include "CoronaLua.h"
#include "utils/LuaEx.h"
#include <Eigen/Eigen>
#include <complex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

//
template<typename T> struct Unmapped {
	//
	using Type = Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using TransType = Eigen::Transpose<Type>;
	using ConstTransType = const Eigen::Transpose<const Type>;

	#ifdef WANT_MAP
		using MappedType = Eigen::Map<Type>;
		using MappedTypeWithInnerStride = Eigen::Map<Type, 0, Eigen::InnerStride<>>;
		using MappedTypeWithOuterStride = Eigen::Map<Type, 0, Eigen::OuterStride<>>;
		using TransMappedType = Eigen::Transpose<MappedType>;
		using TransMappedTypeWithInnerStride = Eigen::Transpose<MappedTypeWithInnerStride>;
		using TransMappedTypeWithOuterStride = Eigen::Transpose<MappedTypeWithOuterStride>;
		using ConstTransMappedType = const Eigen::Transpose<const MappedType>;
		using ConstTransMappedTypeWithInnerStride = const Eigen::Transpose<const MappedTypeWithInnerStride>;
		using ConstTransMappedTypeWithOuterStride = const Eigen::Transpose<const MappedTypeWithOuterStride>;
	#endif

	//
	template<typename U> static const char * AuxSuffix (void) { return ""; }

	template<> static const char * AuxSuffix<TransType> (void) { return "_transpose"; }

	#ifdef WANT_MAP
		template<> static const char * AuxSuffix<MappedType> (void) { return "_map"; }
		template<> static const char * AuxSuffix<MappedTypeWithInnerStride> (void) { return "_map_inner_stride"; }
		template<> static const char * AuxSuffix<MappedTypeWithOuterStride> (void) { return "_map_outer_stride"; }
		template<> static const char * AuxSuffix<TransMappedType> (void) { return "_transposed_map"; }
		template<> static const char * AuxSuffix<TransMappedTypeWithInnerStride> (void) { return "_transposed_map_inner_stride"; }
		template<> static const char * AuxSuffix<TransMappedTypeWithOuterStride> (void) { return "_transposed_map_outer_stride"; }
		template<> static const char * AuxSuffix<ConstTransMappedType>(void) { return "_const_transposed_map"; }
		template<> static const char * AuxSuffix<ConstTransMappedTypeWithInnerStride>(void) { return "_const_transposed_map_inner_stride"; }
		template<> static const char * AuxSuffix<ConstTransMappedTypeWithOuterStride>(void) { return "_const_transposed_map_outer_stride"; }
	#endif

	static const char * Suffix (void) { return AuxSuffix<T>(); }
};

//
template<typename T> struct ScalarName {
	const char * Get (void);
};

template<> inline const char * ScalarName<bool>::Get (void) { return "bool"; }
template<> inline const char * ScalarName<int>::Get (void) { return "int"; }
template<> inline const char * ScalarName<float>::Get (void) { return "float"; }
template<> inline const char * ScalarName<double>::Get (void) { return "double"; }
template<> inline const char * ScalarName<std::complex<float>>::Get (void) { return "cfloat"; }
template<> inline const char * ScalarName<std::complex<double>>::Get (void) { return "cdouble"; }

//
template<typename T> struct Unraveler {
	using Type = T;
};

template<typename T> struct SolverID {
	static const char * Suffix (void) { return ""; }
};

//
template<typename T> const char * FullName (bool bTypeDataKey = false)
{
	using UT = Unraveler<T>::Type;

	static std::string sNameEx;

	sNameEx = "eigen.";

	sNameEx += ScalarName<typename Unmapped<UT>::Type::Scalar>{}.Get();
	sNameEx += Unmapped<UT>::Suffix();

	if (bTypeDataKey) sNameEx += "_TD";

	sNameEx += SolverID<T>::Suffix();

	return sNameEx.c_str();
}


// Per-type data
struct TypeData {
	int mGetAnObjectRef;// Function used to get an instance from the cache, if available
	int mPushRef;	// Function used to push matrix on stack
	int mRegisterObjectRef;	// Function used to add the object to a caching context, when possible
	int mRemoveObjectRef;	// Function used to send the object back to the cache
	int mSelectRef{LUA_NOREF};	// Method used to select some matrix / scalar combination
};

//
template<typename M> void AddTypeData (lua_State * L)
{
	auto td = LuaXS::NewTyped<TypeData>(L);	// ...[, new_type], td
	void * key;

	if (std::is_same<M, BoolMatrix>::value)
	{
		key = td;

		lua_insert(L, -2);	// ..., td, new_type
		lua_pushlightuserdata(L, key);	// ..., td, new_type, key
		lua_insert(L, -2);	// ..., td, key, new_type
		lua_rawset(L, LUA_REGISTRYINDEX);	// ..., td; registry = { ..., [key] = new_type }
	}

	key = GetTypeData<BoolMatrix>(L);

	lua_setfield(L, LUA_REGISTRYINDEX, FullName<M>(true));	// ...; registry = { ..., [eigen.name] = td }
	lua_pushlightuserdata(L, key);	// ..., key
	lua_rawget(L, LUA_REGISTRYINDEX);	// ..., new_type
	lua_pushcfunction(L, LuaXS::TypedGC<M>);// ..., new_type, on_remove
	lua_call(L, 1, 3);	// ..., GetAnObject, RegisterObject, RemoveObject

	td->mRemoveObjectRef = lua_ref(L, 1);	// ..., GetAnObject, RegisterObject
	td->mRegisterObjectRef = lua_ref(L, 1);	// ..., GetAnObject
	td->mGetAnObjectRef = lua_ref(L, 1);// ..., GetAnObject, RegisterObject, RemoveObject
}

//
template<typename T> static TypeData * GetTypeData (lua_State * L)
{
	lua_getfield(L, LUA_REGISTRYINDEX, FullName<T>(true));	// ..., data

	TypeData * td = LuaXS::UD<TypeData>(L, -1);

	lua_pop(L, 1);	// ...

	return td;
}

//
template<typename T, typename R = Unmapped<T>::Type> struct AttachMethods;
template<typename T, typename R = Unmapped<T>::Type> struct AttachSolverMethods;

//
template<typename T> struct Attacher {
	using UT = typename Unraveler<T>::Type;

	template<bool = std::is_same<T, UT>::value> static void Do (lua_State * L)
	{
		AttachMethods<T> am{L};
	}

	template<> static void Do<false> (lua_State * L)
	{
		AttachSolverMethods<T, UT> am{L};
	}
};

//
template<typename T, typename ... Args> T * New (lua_State * L, Args && ... args)
{
	//
	auto td = GetTypeData<T>(L);

	lua_getref(L, td->mGetAnObjectRef);	// ..., GetAnObject
	lua_call(L, 0, 1);	// ..., object?

	T * object = nullptr;

	if (!lua_isnil(L, -1)) object = LuaXS::UD<T>(L, -1);

	else lua_pop(L, 1);	// ...

	if (object) new (object) T(std::forward<Args>(args)...);

	//
	else
	{
		object = LuaXS::NewTyped<T>(L, std::forward<Args>(args)...);	// ..., object

		LuaXS::AttachMethods(L, FullName<T>(), Attacher<T>::Do);
	}

	//
	lua_getref(L, td->mRegisterObjectRef);	// ..., object, RegisterObject
	lua_pushvalue(L, -2);	// ..., object, RegisterObject, object
	lua_call(L, 1, 0);	// ..., object

	return object;
}

//
template<typename T, typename U> int NewMoveRet (lua_State * L, U && m)
{
	New<T>(L, std::move(m));

	return 1;
}

//
template<typename T> int Print (lua_State * L, const T & m)
{
	std::stringstream ss;

	ss << m;

	lua_pushstring(L, ss.str().c_str());// m, str

	return 1;
}

//
template<typename T> void CheckVector (lua_State * L, const T & mat, int arg)
{
	luaL_argcheck(L, mat.cols() == 1 || mat.rows() == 1, arg, "Too few arguments for non-vector");
}

//
template<typename T> T * GetInstance (lua_State * L, int arg = 1)
{
	return LuaXS::CheckUD<T>(L, arg, FullName<T>());
}

//
template<typename T> struct AsVector {
	using Type = Eigen::Map<Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, 1>>;

	static Type To (lua_State * L, int arg = 1)
	{
		T & m = *GetInstance<T>(L, arg);

		CheckVector(L, m, arg);

		return Type{m.data(), m.size()}; // todo: what if this is a map itself?
	}
};

//
#define ADD_INSTANCE_GETTERS()	static T * GetT (lua_State * L, int arg = 1) { return GetInstance<T>(L, arg); }																\
								static R * GetR (lua_State * L, int arg = 1) { return GetInstance<R>(L, arg); }