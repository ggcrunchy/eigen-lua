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
#include <Eigen/Eigen>

//
enum ReductionOption { eDefault, eColwise, eRowwise };

//
inline ReductionOption GetReductionChoice (lua_State * L, int arg)
{
	const char * types[] = { "", "colwise", "rowwise", nullptr };
	ReductionOption opt[] = { eDefault, eColwise, eRowwise };

	return opt[luaL_checkoption(L, arg, "", types)];
}

//
inline int SelfForChaining (lua_State * L)
{
	lua_pushvalue(L, 1);// self, ..., self

	return 1;
}

//
inline bool WantsBool (lua_State * L, const char * str, int arg = -1)
{
	arg = CoronaLuaNormalize(L, arg);

	lua_pushstring(L, str);	// ..., arg, ..., opt

	bool bWants = lua_equal(L, arg, -1) != 0;

	lua_pop(L, 1);	// ..., arg, ...

	return bWants;
}

template<typename T, bool = Eigen::NumTraits<T::Scalar>::IsComplex> struct AuxAsScalar {
	static typename T::Scalar Do (lua_State * L, int arg)
	{
		return Complex<Eigen::NumTraits<T::Scalar>::Real>(L, arg);
	}
};

template<typename T> struct AuxAsScalar<T, false> {
	static typename T::Scalar Do (lua_State * L, int arg)
	{
		return LuaXS::GetArg<T::Scalar>(L, arg);
	}
};

template<typename T> typename T::Scalar AsScalar (lua_State * L, int arg)
{
	return AuxAsScalar<T>::Do(L, arg);
}

//
template<typename T> struct ArgObject {
	T * mMat{nullptr};
	typename T::Scalar mScalar; // complex numbers preclude easy inclusion in union, so don't bother

	ArgObject (lua_State * L, int arg)
	{
		if (HasType<T>(L, arg)) mMat = LuaXS::UD<T>(L, arg);
		else mScalar = AsScalar<T>(L, arg);
	}
};

//
template<typename R> R * SetTemp (lua_State * L, R * temp, int arg)
{
	lua_pushvalue(L, arg);	// ..., other
	luaL_argcheck(L, luaL_getmetafield(L, -1, "asMatrix"), arg, "Type has no conversion method");	// ..., other, asMatrix

	auto td = GetTypeData<R>(L);

	td->mDatum = temp;

	lua_insert(L, -2);	// ..., asMatrix, other
	lua_call(L, 1, 0);	// ...

	td->mDatum = nullptr;

	return temp;
}

//
template<typename R> struct ArgObjectR {
	R mTemp, * mMat{nullptr};
	typename R::Scalar mScalar; // q.v. ArgObject

	ArgObjectR (lua_State * L, int arg)
	{
		if (HasType<R>(L, arg)) mMat = LuaXS::UD<R>(L, arg);
		else if (lua_isuserdata(L, arg)) mMat = SetTemp(L, &mTemp, arg);
		else mScalar = AsScalar<R>(L, arg);
	}
};

//
template<typename T, int R, int C> struct LinSpacing {
	using V = MatrixOf<typename T::Scalar, R, C>;

	template<bool = Eigen::NumTraits<T::Scalar>::IsComplex> static V Make (lua_State * L, int n)
	{
		using RealV = MatrixOf<T::Scalar::value_type, R, C>;

		T::Scalar low = AsScalar<T>(L, 2), high = AsScalar<T>(L, 3);
		V cv;

		cv.resize(n);

		cv.real() = RealV::LinSpaced(n, low.real(), high.real());
		cv.imag() = RealV::LinSpaced(n, low.imag(), high.imag());

		return cv;
	}

	template<> static V Make<false> (lua_State * L, int n)
	{
		return V::LinSpaced(n, AsScalar<T>(L, 2), AsScalar<T>(L, 3));
	}
};

//
template<typename T, typename AO = ArgObject<T>> struct TwoMatrices {
	T mK, * mMat1, * mMat2;
	AO mO1, mO2;

	TwoMatrices (lua_State * L, int arg1 = 1, int arg2 = 2) : mO1{L, arg1}, mO2{L, arg2}
	{
		if (mO1.mMat && mO2.mMat)
		{
			mMat1 = mO1.mMat;
			mMat2 = mO2.mMat;
		}

		else if (mO1.mMat)
		{
			mMat1 = mO1.mMat;
			mMat2 = &mK;

			mK.setConstant(mMat1->rows(), mMat1->cols(), mO2.mScalar);
		}

		else
		{
			mMat1 = &mK;
			mMat2 = mO2.mMat;

			mK.setConstant(mMat2->rows(), mMat2->cols(), mO1.mScalar);
		}
	}
};

template<typename R> using TwoMatricesR = TwoMatrices<R, ArgObjectR<R>>;

//
template<typename T, typename R, typename MM, typename MS, typename SM> R WithMatrixOrScalar (lua_State * L, MM && both, MS && mat_scalar, SM && scalar_mat, int arg1, int arg2)
{
	ArgObject<T> o1{L, arg1};
	ArgObjectR<R> o2{L, arg2};

	if (!o2.mMat) return mat_scalar(*o1.mMat, o2.mScalar);
	else if (!o1.mMat) return scalar_mat(o1.mScalar, *o2.mMat);
	else return both(*o1.mMat, *o2.mMat);
}

//
template<typename R, typename MM, typename MS, typename SM> R WithMatrixOrScalarR (lua_State * L, MM && both, MS && mat_scalar, SM && scalar_mat, int arg1, int arg2)
{
	ArgObjectR<R> o1{L, arg1}, o2{L, arg2};

	if (!o2.mMat) return mat_scalar(*o1.mMat, o2.mScalar);
	else if (!o1.mMat) return scalar_mat(o1.mScalar, *o2.mMat);
	else return both(*o1.mMat, *o2.mMat);
}

//
//
template<typename T, typename R, int RingN = 4> void RingBufferOfMethodThunksProperty (lua_State * L)
{
	New<R>(L);	// ..., meta, temp

	lua_createtable(L, RingN, 1);	// ..., meta, temp, wrappers

	for (int i = 1; i <= RingN; ++i)
	{
		lua_pushvalue(L, -2);	// ..., meta, temp, wrappers, temp
		lua_pushnil(L);	// ..., meta, temp, wrappers, temp, nil
		lua_pushcclosure(L, [](lua_State * L) {
			lua_pushvalue(L, lua_upvalueindex(2));	// obj, ... (args), method
			lua_insert(L, 1);	// method, obj, ...
			lua_pushvalue(L, lua_upvalueindex(1));	// method, obj, ..., temp
			lua_replace(L, 2);	// method, temp, ...
			lua_call(L, lua_gettop(L) - 1, LUA_MULTRET);// ... (results)

			return lua_gettop(L);
		}, 2);	// ..., meta, temp, wrappers, wrapper
		lua_rawseti(L, -2, i);	// ..., meta, temp, wrappers = { ..., wrapper }
	}

	lua_pushinteger(L, 1);	// ..., meta, temp, wrappers, 1
	lua_setfield(L, -2, "pos");	// ..., meta, temp, wrappers = { ..., pos = 1 }

	LuaXS::AttachPropertyParams app;

	app.mUpvalueCount = 2;

	LuaXS::AttachProperties(L, [](lua_State * L) {
		*LuaXS::UD<R>(L, lua_upvalueindex(1)) = *GetInstance<T>(L);

		luaL_getmetafield(L, lua_upvalueindex(1), "__index");	// obj, k, __index
		lua_replace(L, 1);	// __index, k
		lua_rawget(L, 1);	// __index, v?

		if (lua_isfunction(L, 2))
		{
			lua_getfield(L, lua_upvalueindex(2), "pos");// __index, method, ring_pos

			int pos = LuaXS::Int(L, -1);

			lua_pop(L, 1);	// __index, method
			lua_rawgeti(L, lua_upvalueindex(2), pos);	// __index, method, wrapper
			lua_insert(L, 2);	// __index, wrapper, method
			lua_setupvalue(L, 2, 2);// __index, wrapper; wrapper.upval[2] = method
			lua_pushinteger(L, pos % RingN + 1);// __index, method, new_ring_pos
			lua_setfield(L, lua_upvalueindex(2), "pos");// __index, method; wrapper = { ..., pos = new_ring_pos }
		}

		return 1;
	}, app);// ..., meta
}