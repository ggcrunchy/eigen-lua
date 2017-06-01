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
#include "utils.h"
#include <Eigen/Eigen>
#include <bitset>
#include <complex>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

// Alias a few recurring cases.
template<typename T> using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T> using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template<typename T, int O = 0, typename S = Eigen::Stride<0, 0>> using MappedColVector = Eigen::Map<ColVector<T>, O, S>;
template<typename T, int O = 0, typename S = Eigen::Stride<0, 0>> using MappedRowVector = Eigen::Map<RowVector<T>, O, S>;
template<typename S, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic> using MatrixOf = Eigen::Matrix<S, Rows, Cols>;

// Matrices of booleans.
typedef MatrixOf<bool> BoolMatrix;

// Trait to detect expression types, as these often require special handling.
template<typename T> struct IsXpr : std::false_type {};
template<typename U, int R, int C, bool B> struct IsXpr<Eigen::Block<U, R, C, B>> : std::true_type {};
template<typename U, int I> struct IsXpr<Eigen::Diagonal<U, I>> : std::true_type {};
template<typename U> struct IsXpr<Eigen::VectorBlock<U>> : std::true_type {};

// Trait to detect maps.
template<typename T> struct IsMap : std::false_type {};
template<typename U, int O, typename S> struct IsMap<Eigen::Map<U, O, S>> : std::true_type {};

// Trait to detect maps with non-trivial strides.
template<typename T> struct HasNormalStride : std::false_type {};
template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols> struct HasNormalStride<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type {};
template<typename U> struct HasNormalStride<Eigen::Map<U>> : std::true_type {};

// Trait that flags whether some type-related code should be generated (within this module)
// for the matrix family in question.
template<typename R> struct IsMatrixFamilyImplemented : std::false_type {};

// Prefixed to type data registry keys, allowing them to largely reuse the type's own name
// while still playing nicely with luaL_newmetatable(), whose current implementation also
// uses the registry.
#define TYPE_DATA_KEY_SIGNATURE "TD:"

// Various ways to acquire type data that might not yet be present. This is a base type in
// order to avoid dealing with templates for simple lookups, as well as to service any
// logic that must be called in the absence of a known type.
struct GetTypeData {
	enum Info { eIsConvertible, eIsPrimitive, kInfoN };
	enum Options { eDoNothing, eCreateIfMissing, eFetchIfMissing };

	int mSelectRef{LUA_NOREF};	// Method used to select some matrix / scalar combination
	std::bitset<kInfoN> mInfo;	// Some information about the type

	bool GetInfo (Info which) const { return mInfo[which]; }

	// Find a type data given a name on top of the stack.
	static GetTypeData * FromName (lua_State * L)
	{
		lua_pushliteral(L, TYPE_DATA_KEY_SIGNATURE);// ..., name, signature
		lua_insert(L, -2);	// ..., signature, name
		lua_concat(L, 2);	// ..., type_name
		lua_rawget(L, LUA_REGISTRYINDEX);	// ..., type_data?

		GetTypeData * td = LuaXS::UD<GetTypeData>(L, -1);

		lua_pop(L, 1);	// ...

		return td;
	}

	// Find a type data for an Eigen object on the stack.
	static GetTypeData * FromObject (lua_State * L, int arg)
	{
		if (!GetName(L, arg)) return nullptr;	// ...[, name]

		return FromName(L);
	}

	// Get the type name of an Eigen object on the stack.
	static bool GetName (lua_State * L, int arg)
	{
		arg = CoronaLuaNormalize(L, arg);

		if (!luaL_getmetafield(L, arg, "getTypeName")) return false;

		lua_call(L, 0, 1);	// ..., name

		return true;
	}

	// Invoke select logic for the type whose name is on top of the stack.
	static int Select (lua_State * L)
	{
		GetTypeData * td = FromName(L);

		luaL_argcheck(L, td, -1, "Type not part of Eigen library");
		luaL_argcheck(L, td->mSelectRef != LUA_NOREF, -1, "Type does not support select()");
		lua_getref(L, td->mSelectRef);	// bm, then, else[, name], select
		lua_insert(L, 1);	// select, bm, then, else[, name]
		lua_settop(L, 4);	// select, bm, then, else
		lua_call(L, 3, 1);	// selection

		return 1;
	}
};

// Get a given type's data, if present. Since lookup involves building up a name, the pointer
// is saved for subsequent calls. The most common use cases are HasType(), GetInstance(), and
// GetInstanceEx(), where the type will already be loaded on account of an instance existing,
// or in the first case its very absence is enough to answer the question. On account of this,
// the default is to not try to add an absent type. Special options exist to accommodate
// object creation as well as operations like cast(), select(), and complex <-> real functions
// that might cross shared library boundaries and only need to fetch already-loaded type data.
template<typename T> struct InstanceGetter : GetTypeData {
	static T * Get (lua_State * L, Options opts = eDoNothing)
	{
		static ThreadXS::TLS<T *> sThis;

		if (!sThis && (opts == eCreateIfMissing || opts == eFetchIfMissing))
		{
			// Build up the type's name / type data's key. If the library is loaded in parts, e.g. as
			// eigencore + eigendouble, a type might already exist, so this is checked first. Failing
			// that, we move on to actual creation, if desired.
			luaL_Buffer buff;

			luaL_buffinit(L, &buff);
			luaL_addstring(&buff, TYPE_DATA_KEY_SIGNATURE "eigen.");

			AuxTypeName<T::Type>(&buff, L);

			luaL_pushresult(&buff);	// ...[, new_type], name

			const char * name = lua_tostring(L, -1);

			lua_rawget(L, LUA_REGISTRYINDEX);	// ...[, new_type], td?

			sThis = !lua_isnil(L, -1) ? LuaXS::UD<T>(L, -1) : nullptr;

			lua_pop(L, 1);	// ...[, new_type]

			if (!sThis && opts == eCreateIfMissing) sThis = T::Create(L, name);
		}

		return sThis;
	}
};

// Per-type data.
template<typename T, typename R = MatrixOf<typename T::Scalar>> struct TypeData : InstanceGetter<TypeData<T>> {
	typedef T Type;

	const char * mName;	// Cached full name
	int mGetAnObjectRef;// Function used to get an instance from the cache, if available
	int mPushRef{LUA_NOREF};// Function used to push matrix on stack
	int mRegisterObjectRef;	// Function used to add the object to a caching context, when possible
	int mRemoveObjectRef;	// Function used to send the object back to the cache
	int mVectorRingRef{LUA_NOREF};	// Used by vector blocks to maintain a ring of vectors for short term use
	int mWeakListsRef{LUA_NOREF};	// Any weak-keyed tables needed by instances of this type
	void * mDatum{nullptr};	// Pointer to transient datum for some quick operations

	// Get the name of this type, or the key used by its type data.
	const char * GetName (bool bTypeDataKey = false) const
	{
		return bTypeDataKey ? mName : mName + sizeof(TYPE_DATA_KEY_SIGNATURE) - 1;
	}

	// TODO: there could be trouble if the "parent" is reclaimed by the cache but not the dependent object...
	// If the object on the stack is weakly keyed to an item in the given category, pushes it
	// onto the stack; otherwise, pushes nil. (This is useful for "parent"-type connections
	// like maps and views, to keep the original objects alive.)
	int GetRef (lua_State * L, const char * category, int arg) const
	{
		arg = CoronaLuaNormalize(L, arg);

		lua_getref(L, mWeakListsRef);	// ..., arg, ..., weak_lists?

		if (!lua_isnil(L, -1))
		{
			lua_getfield(L, -1, category);	// ..., arg, ..., weak_lists, weak_list?
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

	// Given an object on the stack, weakly keys it in a given category to the item on the top
	// of the stack, popping that. (This would also overwrite any value already so keyed, though
	// thus far all use cases are one-time startup binds.)
	void Ref (lua_State * L, const char * category, int arg)
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
		lua_getfield(L, -1, category);	// ..., arg, ..., arg, item_to_ref, weak_lists, weak_list?

		if (lua_isnil(L, -1))
		{
			lua_pop(L, 1);	// ..., arg, ..., arg, item_to_ref, weak_lists

			LuaXS::NewWeakKeyedTable(L);// ..., arg, ..., arg, item_to_ref, weak_lists, new_weak_list

			lua_pushvalue(L, -1);	// ..., arg, ..., arg, item_to_ref, weak_lists, new_weak_list, new_weak_list
			lua_setfield(L, -3, category);	// ..., arg, ..., arg, item_to_ref, weak_lists = { ..., category = new_weak_list }, new_weak_list
		}

		lua_insert(L, -4);	// ..., arg, ..., weak_list, arg, item_to_ref, weak_lists
		lua_pop(L, 1);	// ..., arg, ..., weak_list, arg, item_to_ref
		lua_rawset(L, -3);	// ..., arg, ..., weak_list = { ..., [arg] = item_item_to_ref }
		lua_pop(L, 1);	// ..., arg, ...
	}

	// Variant of Ref() that looks for the item somewhere on the stack, instead defaulting to
	// the object itself being on top of the stack. The stack is left intact.
	void RefAt (lua_State * L, const char * category, int pos, int arg = -1)
	{
		arg = CoronaLuaNormalize(L, arg);

		lua_pushvalue(L, pos);	// ..., arg, ..., value

		Ref(L, category, arg);	// ..., arg, ...
	}

	// Given an object on the stack, removes the value (if any) to which it is weakly keyed in
	// a given category. (Currently unused.)
	void Unref (lua_State * L, const char * category, int arg)
	{
		if (mWeakListsRef == LUA_NOREF) return;

		lua_pushvalue(L, arg);	// ..., arg, ..., arg
		lua_getref(L, mWeakListsRef);	// ..., arg, ..., arg, weak_lists
		lua_getfield(L, -1, category);	// ..., arg, ..., arg, weak_lists, weak_list?

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

	// Gets the key under which the cache binding logic is stored in the registry. Since boolean
	// matrices are by design part of the core and registered first, their own type data is used
	// as this key. The specialization accounts for this when the BoolMatrix type data itself is
	// created, which is done explicitly rather than lazily via a New(), and expects to pull the
	// aforementioned cache logic off the stack.
	template<typename U = T> static void * GetTypeKey (lua_State * L, TypeData<T> *)
	{
		return TypeData<BoolMatrix>::Get(L, eFetchIfMissing);
	}

	template<> static void * GetTypeKey<BoolMatrix> (lua_State * L, TypeData<T> * td)
	{
		lua_insert(L, -2);	// ..., td, new_type
		lua_pushlightuserdata(L, td);	// ..., td, new_type, key
		lua_insert(L, -2);	// ..., td, key, new_type
		lua_rawset(L, LUA_REGISTRYINDEX);	// ..., td; registry = { ..., [key] = new_type }

		return td;
	}

	// Add push and select functions, allowing interop with code that may be in other modules.
	// The instance in question is converted to a temporary, so these are only available for
	// types where this is possible.
	template<bool = std::is_convertible<T, R>::value> struct AddPushAndSelect {
		AddPushAndSelect (lua_State * L, TypeData<T> * td)
		{
			lua_pushcfunction(L, [](lua_State * L) {
				return NewRet<R>(L, *LuaXS::UD<T>(L, 1));
			});	// meta, push
			lua_pushcfunction(L, [](lua_State * L) {
				auto bm = LuaXS::UD<BoolMatrix>(L, 1); // see the note in BoolMatrix::select()

				return NewRet<R>(L, WithMatrixScalarCombination<R>(L, [bm](const R & m1, const R & m2) {
					return bm->select(m1, m2);
				}, [bm](const R & m, const R::Scalar & s) {
					return bm->select(m, s);
				}, [bm](const R::Scalar & s, const R & m) {
					return bm->select(s, m);
				}, 2, 3));
			});	// meta, push, select

			td->mSelectRef = lua_ref(L, 1);	// ..., push; registry = { ..., ref = select }
			td->mPushRef = lua_ref(L, 1);	// ...; registry = { ..., select, ref = push }
		}
	};

	template<> struct AddPushAndSelect<false> {
		AddPushAndSelect (lua_State *, TypeData<T> *) {}
	};

	// Perform the actual creation, if possible for this sort of type.
	template<bool = IsMatrixFamilyImplemented<R>::value> static TypeData<T> * Create (lua_State * L, const char * name)
	{
		auto td = LuaXS::NewTyped<TypeData<T>>(L);// ...[, new_type], td

		// Register the new type data and fetch the cache binding logic.
		void * key = GetTypeKey<T>(L, td);	// ..., td

		lua_setfield(L, LUA_REGISTRYINDEX, name);	// ...; registry = { ..., [eigen.name] = td }
		lua_pushlightuserdata(L, key);	// ..., key
		lua_rawget(L, LUA_REGISTRYINDEX);	// ..., new_type

		// Wire the new type into the cache, supplying some callbacks. The first of these is
		// called if the instance is reclaimed: any associations made by Ref() are cleared.
		lua_createtable(L, 0, 2);	// ..., new_type, opts
		lua_pushcfunction(L, [](lua_State * L) {
			auto td = Get(L);

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

		// Secondly, when the object is fetched from the cache, its current contents are destructed.
		// This is done here rather than when caching to avoid problems during lua_close().
		// TODO: for types like dynamic matrices this could be done in on_cache, by calling the destructor here and then
		// placement new'ing a null matrix, likewise with maps...
		lua_pushcfunction(L, LuaXS::TypedGC<T>);// ..., new_type, opts, on_fetch
		lua_setfield(L, -2, "on_fetch");// ..., new_type, opts = { on_cache, on_fetch = on_fetch }

		// Bind the type to the cache.
		lua_call(L, 1, 3);	// ..., GetAnObject, RegisterObject, RemoveObject

		td->mRemoveObjectRef = lua_ref(L, 1);	// ..., GetAnObject, RegisterObject
		td->mRegisterObjectRef = lua_ref(L, 1);	// ..., GetAnObject
		td->mGetAnObjectRef = lua_ref(L, 1);// ...

		// Hook up routines to push a matrix, e.g. from another shared library, and with similar
		// reasoning to use such matrices in BoolMatrix::select().
		AddPushAndSelect<> apas{L, td};

		// Capture some information needed when the exact type is unknown.
		td->mInfo[eIsConvertible] = std::is_convertible<T, R>::value;
		td->mInfo[eIsPrimitive] = std::is_same<T, R>::value;

		// Save the name for use down the road.
		lua_pushstring(L, name);// ..., interned_name

		td->mName = lua_tostring(L, -1);

		lua_pop(L, 1);	// ...

		return td;
	}

	template<> static TypeData<T> * Create<false> (lua_State * L, const char * name)
	{
		luaL_error(L, "Unable to implement %s\n", name);

		return nullptr;
	}
};

// Does the object on the stack have the given type?
template<typename T> bool HasType (lua_State * L, int arg)
{
	auto td = TypeData<T>::Get(L);

	return td != nullptr && LuaXS::IsType(L, td->GetName(), arg);
}

// Helper to convert an object to a matrix, say for an "asMatrix" method. This also has a
// special case for internal use that populates a stack-based temporary.
template<typename T, typename R> int AsMatrix (lua_State * L)
{
	auto td = TypeData<R>::Get(L);
	T & object = *GetInstance<T>(L);

	if (!td->mDatum) return NewRet<R>(L, object);

	else
	{
		*static_cast<R *>(td->mDatum) = object;

		return 0;
	}
}

// This is specialized to populate a given type's methods and properties.
template<typename T, typename R = MatrixOf<typename T::Scalar>> struct AttachMethods {
	AttachMethods (lua_State * L)
	{
		luaL_error(L, "Unsupported type: %s", TypeName<T>(L));
	}
};

// Instantiate a type, rigging up the type itself on the first call.
template<typename T, typename ... Args> T * New (lua_State * L, Args && ... args)
{
	// Try to fetch an instance from the cache. If found, reuse its memory.
	auto td = TypeData<T>::Get(L, GetTypeData::eCreateIfMissing);

	lua_getref(L, td->mGetAnObjectRef);	// ..., GetAnObject
	lua_call(L, 0, 1);	// ..., object?

	T * object = nullptr;

	if (!lua_isnil(L, -1)) object = LuaXS::UD<T>(L, -1);

	else lua_pop(L, 1);	// ...

	if (object) new (object) T(std::forward<Args>(args)...);

	// Otherwise, add a new object. If the type itself is new, attach its methods as well.
	else
	{
		object = LuaXS::NewTyped<T>(L, std::forward<Args>(args)...);// ..., object

		LuaXS::AttachMethods(L, td->GetName(), [](lua_State * L) {
			AttachMethods<T> am{L};

			// Add a type name getter method, allowing for interal queries for the type key.
			lua_pushstring(L, TypeData<T>::Get(L)->GetName());	// ..., meta, name
			lua_pushcclosure(L, [](lua_State * L) {
				lua_pushvalue(L, lua_upvalueindex(1));	// [object, ]name

				return 1;
			}, 1);	// ..., meta, GetTypeName
			lua_setfield(L, -2, "getTypeName");	// ..., meta = { ..., getTypeName = GetTypeName }
		});
	}

	// If caching is active, register the object for later reclamation.
	lua_getref(L, td->mRegisterObjectRef);	// ..., object, RegisterObject
	lua_pushvalue(L, -2);	// ..., object, RegisterObject, object
	lua_call(L, 1, 0);	// ..., object

	return object;
}

// Helper for the common case where a new instance is immediately returned.
template<typename T, typename U> int NewRet (lua_State * L, U && m)
{
	New<T>(L, std::forward<U>(m));

	return 1;
}

// Common forms for instantiating dependent types.
#define NEW_REF1_NO_RET(T, KEY, INPUT)	New<T>(L, INPUT);	/* source, ..., new_object */	\
										TypeData<T>::Get(L)->RefAt(L, KEY, 1)
#define NEW_REF1(T, KEY, INPUT)	NEW_REF1_NO_RET(T, KEY, INPUT);	/* source, ..., new_object */	\
																								\
								return 1
#define NEW_REF1_DECLTYPE(KEY, INPUT) NEW_REF1(decltype(INPUT), KEY, INPUT) /* source, ..., new_object */
#define NEW_REF1_DECLTYPE_MOVE(KEY, INPUT) NEW_REF1(decltype(INPUT), KEY, std::move(INPUT)) /* source, ..., new_object */

// Convert a matrix to a pretty-printed string, e.g. for use by __tostring.
template<typename T> int Print (lua_State * L, const T & m)
{
	std::stringstream ss;

	ss << m;

	lua_pushstring(L, ss.str().c_str());// m, str

	return 1;
}

// Helper to ensure a matrix has vector shape.
template<typename T> void CheckVector (lua_State * L, const T & mat, int arg)
{
	luaL_argcheck(L, mat.cols() == 1 || mat.rows() == 1, arg, "Non-vector: row and column counts both exceed 1");
}

// Structure a matrix or similar object as a vector via Eigen's Ref type.
template<typename R, bool bRow> struct VectorRef {
	using V = typename std::conditional<bRow,
		typename RowVector<typename R::Scalar>,
		typename ColVector<typename R::Scalar>
	>::type;
	using Type = typename std::conditional<bRow,
		Eigen::Ref<typename V, 0, Eigen::InnerStride<>>,
		Eigen::Ref<typename V>
	>::type;

	std::unique_ptr<Type> mRef;	// Reference to an object to interpret as a vector
	R mTemp;// Temporary in cases of row vectors or complex types
	bool mChanged;	// Did we have to resort to a temporary?
	bool mTransposed;	// Did we also need to transpose that temporary?

	bool Changed (void) const { return mChanged; }

	// Add some operators rather than methods for convenient Ref access.
	const Type & operator * (void)
	{
		return *mRef;
	}

	Type * operator -> (void)
	{
		return mRef.get();
	}

	// When returning or updating a vector, we generally want the end result to have the same
	// shape as the source, even though it was convenient to work in one form. This will 
	void RestoreShape (R * object = nullptr)
	{
		if (!object) object = &mTemp;
		if (mTransposed) object->transposeInPlace();
	}

	// Allow stashing in data structures.
	VectorRef (void) = default;

	// Does the object have the shape appropriate to the vector type?
	template<bool = bRow> bool ShapeIsCorrect (R * object) const
	{
		return object->rows() == 1;
	}

	template<> bool ShapeIsCorrect<false> (R * object) const
	{
		return object->cols() == 1;
	}

	// Attach the appropriately-typed vector object to the reference.
	template<bool = bRow> void Bind (R * object)
	{
		mRef.reset(new Type{object->row(0)});
	}

	template<> void Bind<false> (R * object)
	{
		mRef.reset(new Type{object->col(0)});
	}

	// Bind a vector, possibly via a temporary. This is separate from the constructor for
	// cases when these are inside other data structures.
	void Init (lua_State * L, int arg = 1)
	{
		GetTypeData * td = GetTypeData::FromObject(L, arg);

		luaL_argcheck(L, td && td->GetInfo(GetTypeData::eIsConvertible), arg, "Not a convertible Eigen object");

		R * ptr = td->GetInfo(GetTypeData::eIsPrimitive) ? GetInstance<R>(L, arg) : nullptr;

		mChanged = mTransposed = false;

		if (!ptr || !ShapeIsCorrect(ptr))
		{
			mTemp = ptr ? *ptr : GetInstanceEx<R>(L, arg);
			ptr = &mTemp;

			if (!ShapeIsCorrect(&mTemp))
			{
				mTemp.transposeInPlace();

				mTransposed = true;
			}

			luaL_argcheck(L, ShapeIsCorrect(&mTemp), arg, "Non-vector: row and column counts both exceed 1");
		}

		Bind(ptr);
	}

	VectorRef (lua_State * L, int arg = 1)
	{
		Init(L, arg);
	}
};

// Column vectors are used pervasively, so alias them.
template<typename R> using ColumnVector = VectorRef<R, false>;

// Acquire an instance whose exact type is expected.
template<typename T> T * GetInstance (lua_State * L, int arg = 1)
{
	auto td = TypeData<T>::Get(L);

	luaL_argcheck(L, td, arg, "No such type");

	return LuaXS::CheckUD<T>(L, arg, td->GetName());
}

// Acquire an instance resolved to a matrix type, with shortcuts for common source types.
template<typename R> R GetInstanceEx (lua_State * L, int arg = 1)
{
	if (HasType<R>(L, arg)) return *LuaXS::UD<R>(L, arg);

	R temp;

	if (HasType<Eigen::Block<R>>(L, arg)) temp = *LuaXS::UD<Eigen::Block<R>>(L, arg);
	else if (HasType<Eigen::Transpose<R>>(L, arg)) temp = *LuaXS::UD<Eigen::Transpose<R>>(L, arg);
	else if (HasType<Eigen::Block<Eigen::Transpose<R>>>(L, arg)) temp = *LuaXS::UD<Eigen::Block<Eigen::Transpose<R>>>(L, arg);
	else SetTemp(L, &temp, arg);

	return temp;
}

// Standard boilerplate for getters. Aside from the case of binary metamethods, the first
// type in a given method will be a "self" whose type we can assume, and thus we return a
// pointer to an instance of that very type. Other arguments are generally given some leeway,
// since they might be blocks, maps, transposes, etc. In these cases, the object is copied
// to a new matrix (of some commonly resolved type) that gets returned; this is potentially
// expensive, but offers flexibility. (The "R" originally stood for "return value", though
// "raw", "right-hand side", and "resolved" tend to fit as well.)
template<typename T, typename R> struct InstanceGetters {
	static T * GetT (lua_State * L, int arg = 1) { return GetInstance<T>(L, arg); }
	static R GetR (lua_State * L, int arg = 1) { return GetInstanceEx<R>(L, arg); }
};

// The following builds up a name piecemeal by unraveling important pieces of the type. In
// theory <typeinfo> might do as well, but quoting http://en.cppreference.com/w/cpp/types/type_info/name,
// "No guarantees are given, in particular, the returned string can be identical for several
// types and change between invocations of the same program.", which is especially relevant
// since the implementation may be spread among two or more shared libraries (with a common
// Lua state), e.g. eigencore + eigendouble.
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

		// ^^^ TODO: options, max_rows, max_cols, if different from defaults

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

//
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