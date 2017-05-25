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
#include "utils/Thread.h"
#include <Eigen/Eigen>
#include <complex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

//
template<typename T> using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T> using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template<typename T, int O = 0, typename S = Eigen::Stride<0, 0>> using MappedColVector = Eigen::Map<ColVector<T>, O, S>;
template<typename T, int O = 0, typename S = Eigen::Stride<0, 0>> using MappedRowVector = Eigen::Map<RowVector<T>, O, S>;

//
template<typename S, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic> using MatrixOf = Eigen::Matrix<S, Rows, Cols>;

template<typename T> struct RawMatrix {
	using Type = MatrixOf<typename T::Scalar>;
};

//
template<typename T> struct IsStrideFree : std::false_type {};
template<typename T, int Rows, int Cols> struct IsStrideFree<Eigen::Matrix<T, Rows, Cols>> : std::true_type {};
template<typename U> struct IsStrideFree<Eigen::Map<U>> : std_true_type {};

//
#define TYPE_DATA_KEY_SIGNATURE "TD:"

// Per-type data
struct TypeData {
	const char * mName;	// Cached full name
	int mGetAnObjectRef;// Function used to get an instance from the cache, if available
	int mPushRef;	// Function used to push matrix on stack
	int mRegisterObjectRef;	// Function used to add the object to a caching context, when possible
	int mRemoveObjectRef;	// Function used to send the object back to the cache
	int mSelectRef{LUA_NOREF};	// Method used to select some matrix / scalar combination
	int mVectorRingRef{LUA_NOREF};	// Used by vector blocks to maintain a ring of quick-useAc vectors
	int mWeakListsRef{LUA_NOREF};	// Any weak-keyed tables needed by instances of this type
	void * mDatum{nullptr};	// Pointer to transient datum for some quick operations

	//
	const char * GetName (bool bTypeDataKey = false) const
	{
		return bTypeDataKey ? mName : mName + sizeof(TYPE_DATA_KEY_SIGNATURE) - 1;
	}

	//
	int GetRef (lua_State * L, const char * key, int arg) const
	{
		arg = CoronaLuaNormalize(L, arg);

		lua_getref(L, mWeakListsRef);	// ..., arg, ..., weak_lists?

		if (!lua_isnil(L, -1))
		{
			lua_getfield(L, -1, key);	// ..., arg, ..., weak_lists, weak_list?
			lua_remove(L, -2);	// ..., arg, ..., weak_list?

			if (!lua_isnil(L, -1))
			{
				lua_pushvalue(L, arg);	// ..., arg, ..., weak_list, arg
				lua_rawget(L, -2);	// ..., arg, ..., weak_list, value
				lua_replace(L, -2);	// ..., arg, ..., value
			}
		}

		return 1;
	}

	//
	void Ref (lua_State * L, const char * key, int arg)
	{
		luaL_argcheck(L, arg != -1 && arg != lua_gettop(L), arg, "Attempt to ref self");

		if (mWeakListsRef == LUA_NOREF)
		{
			lua_newtable(L);// ..., arg, ..., item_to_ref, lists

			mWeakListsRef = lua_ref(L, 1);	// ..., arg, ..., item_to_ref
		}

		lua_pushvalue(L, arg);	// ..., arg, ..., item_to_ref, arg
		lua_insert(L, -2);	// ..., arg, ..., arg, item_to_ref
		lua_getref(L, mWeakListsRef);	// ..., arg, ..., arg, item_to_ref, weak_lists
		lua_getfield(L, -1, key);	// ..., arg, ..., arg, item_to_ref, weak_lists, weak_list?

		if (lua_isnil(L, -1))
		{
			lua_pop(L, 1);	// ..., arg, ..., arg, item_to_ref, weak_lists

			LuaXS::NewWeakKeyedTable(L);// ..., arg, ..., arg, item_to_ref, weak_lists, new_weak_list

			lua_pushvalue(L, -1);	// ..., arg, ..., arg, item_to_ref, weak_lists, new_weak_list, new_weak_list
			lua_setfield(L, -3, key);	// ..., arg, ..., arg, item_to_ref, weak_lists = { ..., key = new_weak_list }, new_weak_list
		}

		lua_insert(L, -4);	// ..., arg, ..., weak_list, arg, item_to_ref, weak_lists
		lua_pop(L, 1);	// ..., arg, ..., weak_list, arg, item_to_ref
		lua_rawset(L, -3);	// ..., arg, ..., weak_list = { ..., [arg] = item_item_to_ref }
		lua_pop(L, 1);	// ..., arg, ...
	}

	//
	void RefAt (lua_State * L, const char * key, int pos, int arg = -1)
	{
		arg = CoronaLuaNormalize(L, arg);

		lua_pushvalue(L, pos);	// ..., arg, ..., value

		Ref(L, key, arg);	// ..., arg, ...
	}

	//
	void Unref (lua_State * L, const char * key, int arg)
	{
		if (mWeakListsRef == LUA_NOREF) return;

		lua_pushvalue(L, arg);	// ..., arg, ..., arg
		lua_getref(L, mWeakListsRef);	// ..., arg, ..., arg, weak_lists
		lua_getfield(L, -1, key);	// ..., arg, ..., arg, weak_lists, weak_list?

		if (lua_isnil(L, -1)) lua_pop(L, 3);// ..., arg, ...

		else
		{
			lua_insert(L, -3);	// ..., arg, ..., weak_list, arg, weak_lists
			lua_pop(L, 1);	// ..., arg, ..., weak_list, arg
			lua_pushnil(L);	// ..., arg, ..., weak_list, arg, nil
			lua_rawset(L, -3);	// ..., arg, ..., weak_list = { ..., [arg] = nil }
			lua_pop(L, 1);	// ..., arg, ...
		}
	}
};

//
typedef MatrixOf<bool> BoolMatrix;

//
template<typename T> static void * GetTypeKey (lua_State * L, TypeData *)
{
	return GetTypeData<BoolMatrix>(L, true);
}

template<> static void * GetTypeKey<BoolMatrix> (lua_State * L, TypeData * td)
{
	lua_insert(L, -2);	// ..., td, new_type
	lua_pushlightuserdata(L, td);	// ..., td, new_type, key
	lua_insert(L, -2);	// ..., td, key, new_type
	lua_rawset(L, LUA_REGISTRYINDEX);	// ..., td; registry = { ..., [key] = new_type }

	return td;
}

//
template<typename T> static TypeData * AddTypeData (lua_State * L)
{
	luaL_Buffer buff;
	
	luaL_buffinit(L, &buff);
	luaL_addstring(&buff, TYPE_DATA_KEY_SIGNATURE "eigen.");

	AuxTypeName<T>(&buff, L);

	luaL_pushresult(&buff);	// ...[, new_type], name

	const char * name = lua_tostring(L, -1);

	lua_rawget(L, LUA_REGISTRYINDEX);	// ...[, new_type], td?

	TypeData * td = !lua_isnil(L, -1) ? LuaXS::UD<TypeData>(L, -1) : nullptr;

	lua_pop(L, 1);	// ...[, new_type]

	if (!td)
	{
		td = LuaXS::NewTyped<TypeData>(L);	// ...[, new_type], td

		void * key = GetTypeKey<T>(L, td);	// ..., td

		lua_setfield(L, LUA_REGISTRYINDEX, name);	// ...; registry = { ..., [eigen.name] = td }
		lua_pushlightuserdata(L, key);	// ..., key
		lua_rawget(L, LUA_REGISTRYINDEX);	// ..., new_type
		lua_createtable(L, 0, 2);	// ..., new_type, opts
		lua_pushcfunction(L, [](lua_State * L) {
			auto td = GetTypeData<T>(L);

			lua_getref(L, td->mWeakListsRef);	// object, weak_lists?
			
			if (!lua_isnil(L, -1))
			{
				for (lua_pushnil(L); lua_next(L, -2); lua_pop(L, 1))
				{
					lua_pushvalue(L, 1);// object, weak_lists, key, weak_list, object
					lua_pushnil(L);	// object, weak_lists, key, weak_list, object, nil
					lua_rawset(L, -3);	// object, weak_lists, key, weak_list = { ..., [object] = nil }
				}
			}

			return 0;
		});	// ..., new_type, opts, on_cache
		lua_setfield(L, -2, "on_cache");// ..., new_type, opts = { on_cache = on_cache }
		lua_pushcfunction(L, LuaXS::TypedGC<T>);// ..., new_type, opts, on_fetch
		lua_setfield(L, -2, "on_fetch");// ..., new_type, opts = { on_cache, on_fetch = on_fetch }
		lua_call(L, 1, 3);	// ..., GetAnObject, RegisterObject, RemoveObject
		lua_pushstring(L, name);// ..., GetAnObject, RegisterObject, RemoveObject, interned_name

		td->mName = lua_tostring(L, -1);

		lua_pop(L, 1);	// ..., GetAnObject, RegisterObject, RemoveObject

		td->mRemoveObjectRef = lua_ref(L, 1);	// ..., GetAnObject, RegisterObject
		td->mRegisterObjectRef = lua_ref(L, 1);	// ..., GetAnObject
		td->mGetAnObjectRef = lua_ref(L, 1);// ...
	}

	return td;
}

//
template<typename T> TypeData * GetTypeData (lua_State * L, bool bAddIfAbsent = false)
{
	static ThreadXS::TLS<TypeData *> sThisType;

	if (!sThisType && bAddIfAbsent) sThisType = AddTypeData<T>(L);

	return sThisType;
}

//
template<typename T> bool HasType (lua_State * L, int arg)
{
	auto td = GetTypeData<T>(L);

	return td != nullptr && LuaXS::IsType(L, td->GetName(), arg);
}

//
template<typename T, typename R> int AsMatrix (lua_State * L)
{
	auto td = GetTypeData<R>(L);
	T & object = *GetInstance<T>(L);

	if (!td->mDatum) return NewRet<R>(L, object);

	else
	{
		*static_cast<R *>(td->mDatum) = object;

		return 0;
	}
}

//
template<typename T> struct IsXpr : std::false_type {};

template<typename U, int R, int C, bool B> struct IsXpr<Eigen::Block<U, R, C, B>> : std::true_type {};
template<typename U, int I> struct IsXpr<Eigen::Diagonal<U, I>> : std::true_type {};
template<typename U> struct IsXpr<Eigen::VectorBlock<U>> : std::true_type {};

//
template<typename T, typename R = RawMatrix<T>::Type> struct AttachMethods {
	AttachMethods (lua_State * L)
	{
		luaL_error(L, "Unsupported type: %s", TypeName<T>(L));
	}
};

//
template<typename T, typename ... Args> T * New (lua_State * L, Args && ... args)
{
	//
	auto td = GetTypeData<T>(L, true);

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

		LuaXS::AttachMethods(L, td->GetName(), [](lua_State * L) {
			AttachMethods<T> am{L};
		});
	}

	//
	lua_getref(L, td->mRegisterObjectRef);	// ..., object, RegisterObject
	lua_pushvalue(L, -2);	// ..., object, RegisterObject, object
	lua_call(L, 1, 0);	// ..., object

	return object;
}

//
template<typename T, typename U> int NewRet (lua_State * L, U && m)
{
	New<T>(L, std::forward<U>(m));

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
	luaL_argcheck(L, mat.cols() == 1 || mat.rows() == 1, arg, "Non-vector: row and column counts both exceed 1");
}

//
template<typename T> T * GetInstance (lua_State * L, int arg = 1)
{
	auto td = GetTypeData<T>(L);

	luaL_argcheck(L, td, arg, "No such type");

	return LuaXS::CheckUD<T>(L, arg, td->GetName());
}

//
template<typename T, bool = IsStrideFree<T>::value> struct AsVector {
	using Type = Eigen::Map<ColVector<typename T::Scalar>>;

	static void New (Type * v, T * m)
	{
		new (v) Type{m->data(), m->size()};
	}

	static Type To (T * m)
	{
		return Type{m->data(), m->size()};
	}

	static Type To (lua_State * L, int arg = 1)
	{
		T * m = GetInstance<T>(L, arg);

		CheckVector(L, *m, arg);

		return To(m); 
	}
};

// TODO: will this work?
template<typename U, int O, typename S> struct AsVector<Eigen::Map<U, O, S>, false> {
	using Type = Eigen::Map<ColVector<typename U::Scalar>, O, S>;
	using T = Eigen::Map<U, O, S>;

	static void New (Type * v, T * m)
	{
		new (v) Type{m->data(), m->size(), S{m->outerStride(), m->innerStride()}};
	}

	static Type To (T * m)
	{
		return Type{m->data(), m->size(), S{m->outerStride(), m->innerStride()}};
	}

	static Type To (lua_State * L, int arg = 1)
	{
		T * m = GetInstance<T>(L, arg);

		CheckVector(L, *m, arg);

		return To(m); 
	}
};

//
template<typename R> R GetREx (lua_State * L, int arg = 1)
{
	if (HasType<R>(L, arg)) return *LuaXS::UD<R>(L, arg);

	R temp;

	if (HasType<Eigen::Block<R>>(L, arg)) temp = *LuaXS::UD<Eigen::Block<R>>(L, arg);
	else if (HasType<Eigen::Transpose<R>>(L, arg)) temp = *LuaXS::UD<Eigen::Transpose<R>>(L, arg);
	else if (HasType<Eigen::Block<Eigen::Transpose<R>>>(L, arg)) temp = *LuaXS::UD<Eigen::Block<Eigen::Transpose<R>>>(L, arg);
	else SetTemp(L, &temp, arg);

	return temp;
}

//
#define ADD_INSTANCE_GETTERS()	static T * GetT (lua_State * L, int arg = 1) { return GetInstance<T>(L, arg); }	\
								static R GetR (lua_State * L, int arg = 1) { return GetREx<R>(L, arg); }


template<typename T> struct AuxTypeName {
	AuxTypeName (luaL_Buffer *, lua_State * L)
	{
		luaL_error(L, "Unsupported type");
	}
};

template<typename T> struct AuxTypeName<const T> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "const ");

		AuxTypeName<T> atn{B, L};
	}
};

template<> struct AuxTypeName<bool> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, "bool"); }
};

template<> struct AuxTypeName<int> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, "int"); }
};

template<> struct AuxTypeName<float> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, "float"); }
};

template<> struct AuxTypeName<double> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, "double"); }
};

template<> struct AuxTypeName<std::complex<float>> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, "cfloat"); }
};

template<> struct AuxTypeName<std::complex<double>> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, "cdouble"); }
};

void AddDynamicOrN (luaL_Buffer * B, lua_State * L, int n)
{
	if (n == Eigen::Dynamic) luaL_addstring(B, "dynamic");

	else
	{
		lua_pushfstring(L, "%d", n);// ..., n
		luaL_addvalue(B);	// ...
	}
}

template<typename Scalar, int R, int C, int O, int MR, int MC> struct AuxTypeName<Eigen::Matrix<Scalar, R, C, O, MR, MC>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		AuxTypeName<Scalar>(B, L);

		if (R == 1 || C == 1)
		{
			lua_pushfstring(L, "_%s_vector[", R == 1 ? "row" : "col");	// ..., what
			luaL_addvalue(B);	// ...

			AddDynamicOrN(B, L, R == 1 ? C : R);
		}

		else
		{
			luaL_addstring(B, "_matrix[");

			AddDynamicOrN(B, L, R);

			luaL_addstring(B, ", ");

			AddDynamicOrN(B, L, C);
		}

		luaL_addstring(B, "]");
	}
};

template<> struct AuxTypeName<Eigen::Stride<0, 0>> {
	AuxTypeName (luaL_Buffer * B, lua_State *) {}
};

template<> struct AuxTypeName<Eigen::InnerStride<>> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, ", InnerStride"); }
};

template<> struct AuxTypeName<Eigen::OuterStride<>> {
	AuxTypeName (luaL_Buffer * B, lua_State *) { luaL_addstring(B, ", OuterStride"); }
};

template<typename U, int O, typename S> struct AuxTypeName<Eigen::Map<U, O, S>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "Map<");

		AuxTypeName<U>(B, L);

		lua_pushfstring(L, ", %d", O);// ..., options
		luaL_addvalue(B);	// ...

		AuxTypeName<S>(B, L);

		luaL_addstring(B, ">");
	}
};

template<typename U> struct AuxTypeName<Eigen::Transpose<U>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "Transpose<");

		AuxTypeName<U>(B, L);

		luaL_addstring(B, ">");
	}
};

template<typename T> const char * TypeName (lua_State * L)
{
	luaL_Buffer B;

	luaL_buffinit(L, &B);

	AuxTypeName<T>(&B, L);

	luaL_pushresult(&B);// ..., str

	return lua_tostring(L, -1);
}