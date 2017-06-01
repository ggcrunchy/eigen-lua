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
#include "ByteReader.h"
#include <Eigen/Eigen>
#include <complex>

// Helper to return the instance again for method chaining.
inline int SelfForChaining (lua_State * L)
{
	lua_pushvalue(L, 1);// self, ..., self

	return 1;
}

// Check for strings whose presence means true.
inline bool WantsBool (lua_State * L, const char * str, int arg = -1)
{
	arg = CoronaLuaNormalize(L, arg);

	lua_pushstring(L, str);	// ..., arg, ..., opt

	bool bWants = lua_equal(L, arg, -1) != 0;

	lua_pop(L, 1);	// ..., arg, ...

	return bWants;
}

// Helper to fetch a coefficient from a matrix.
template<typename T> int Call (lua_State * L)
{
	const T & m = *GetInstance<T>(L);
	int a = LuaXS::Int(L, 2) - 1;
	T::Scalar result;

	if (lua_gettop(L) == 2)
	{
		CheckVector(L, m, 1);

		result = m.cols() == 1 ? m(a, 0) : m(0, a);
	}

	else result = m(a, LuaXS::Int(L, 3) - 1);

	return LuaXS::PushArgAndReturn(L, result);
}

// Helper to read complex numbers in various formats.
template<typename T> static std::complex<T> Complex (lua_State * L, int arg)
{
	switch (lua_type(L, arg))
	{
	case LUA_TNUMBER:
		return std::complex<T>{static_cast<T>(lua_tonumber(L, arg)), static_cast<T>(0)};
	case LUA_TTABLE:
		arg = CoronaLuaNormalize(L, arg);

		lua_rawgeti(L, arg, 1);	// ..., complex, ..., r
		lua_rawgeti(L, arg, 2);	// ..., complex, ..., r, i

		{
			double r = luaL_optnumber(L, -2, 0.0), i = luaL_optnumber(L, -1, 0.0);

			lua_pop(L, 2);	// ..., complex, ...

			return std::complex<T>{static_cast<T>(r), static_cast<T>(i)};
		}
	default:
		ByteReader reader{L, arg};

		luaL_argcheck(L, reader.mBytes && reader.mCount >= sizeof(std::complex<T>), arg, "Invalid complex number");

		std::complex<T> comp;

		memcpy(&comp, reader.mBytes, sizeof(std::complex<T>));

		return comp;
	}
}

// Helper to reduce an object given some user-supplied function.
template<typename T, typename R, typename RRT> RRT Redux (lua_State * L)
{
	auto func = [L](const T::Scalar & x, const T::Scalar & y)
	{
		LuaXS::PushMultipleArgs(L, LuaXS::StackIndex{L, 2}, x, y);	// mat, func[, how], func, x, y

		lua_call(L, 2, 1);	// mat, func[, how], result

		T::Scalar result(0);

		if (!lua_isnil(L, -1)) result = AsScalar<R>(L, -1);

		lua_pop(L, 1);	// mat, func

		return result;
	};

	return GetInstance<T>(L)->redux(func);
}

// Specialize PushArg() for complex types to streamline code elsewhere, e.g. in macros.
template<> inline void LuaXS::PushArg<std::complex<double>> (lua_State * L, std::complex<double> c)
{
	lua_createtable(L, 2, 0);	// ..., c
	lua_pushnumber(L, c.real());// ..., c, c.r
	lua_rawseti(L, -2, 1);	// ..., c = { r }
	lua_pushnumber(L, c.imag());// ..., c, c.i
	lua_rawseti(L, -2, 2);	// ..., c = { r, i }
}

template<> inline void LuaXS::PushArg<std::complex<float>> (lua_State * L, std::complex<float> c)
{
	LuaXS::PushArg(L, std::complex<double>{c.real(), c.imag()});
}

// Helper to fetch a scalar from the stack.
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

// Object that fetches an item from the stack that may be either an object instance or scalar.
template<typename T> struct ArgObject {
	T * mObject{nullptr};
	typename T::Scalar mScalar; // scalar may be complex, so unions would have been messy

	ArgObject (lua_State * L, int arg)
	{
		if (HasType<T>(L, arg)) mObject = LuaXS::UD<T>(L, arg);
		else mScalar = AsScalar<T>(L, arg);
	}
};

// Populate a matrix (typically a stack temporary) by going through an object's "asMatrix"
// method. These methods are all built on AsMatrix(), which special cases the logic here.
template<typename R> R * SetTemp (lua_State * L, R * temp, int arg)
{
	lua_pushvalue(L, arg);	// ..., other
	luaL_argcheck(L, luaL_getmetafield(L, -1, "asMatrix"), arg, "Type has no conversion method");	// ..., other, asMatrix

	auto td = TypeData<R>::Get(L);

	td->mDatum = temp;

	lua_insert(L, -2);	// ..., asMatrix, other
	lua_call(L, 1, 0);	// ...

	td->mDatum = nullptr;

	return temp;
}

// Special case of ArgObject where the object case resolves to a matrix.
template<typename R> struct ArgObjectR {
	R mTemp, * mObject{nullptr}; // "object" for conformity with ArgObject
	typename R::Scalar mScalar; // q.v. ArgObject

	ArgObjectR (lua_State * L, int arg)
	{
		if (HasType<R>(L, arg)) mObject = LuaXS::UD<R>(L, arg);
		else if (lua_isuserdata(L, arg)) mObject = SetTemp(L, &mTemp, arg); // TODO: better predicate? scalar could have userdata __bytes
		else mScalar = AsScalar<R>(L, arg);
	}
};

// Gets two matrices from items on the stack, where at least one is assumed to resolve to
// a matrix type. Scalar items are converted to constant matrices, whereas otherwise the
// ArgObjectR logic is applied.
template<typename R> struct TwoMatrices {
	R mK, * mMat1, * mMat2;
	ArgObjectR<R> mO1, mO2;

	static void CheckTwo (lua_State * L, bool bOK, int arg1, int arg2)
	{
		if (!bOK) luaL_error(L, "At least one of arguments %i and %i must resolve to a matrix", arg1, arg2);
	}

	TwoMatrices (lua_State * L, int arg1 = 1, int arg2 = 2) : mO1{L, arg1}, mO2{L, arg2}
	{
		if (mO1.mObject && mO2.mObject)
		{
			mMat1 = mO1.mObject;
			mMat2 = mO2.mObject;
		}

		else if (mO1.mObject)
		{
			mMat1 = mO1.mObject;
			mMat2 = &mK;

			mK.setConstant(mMat1->rows(), mMat1->cols(), mO2.mScalar);
		}

		else
		{
			CheckTwo(L, mO2.mObject != nullptr, arg1, arg2);

			mMat1 = &mK;
			mMat2 = mO2.mObject;

			mK.setConstant(mMat2->rows(), mMat2->cols(), mO1.mScalar);
		}
	}
};

// Perform some binary operation from items on the stack where at least one of them is
// assumed to resolve to a matrix type. Scalars are left as is, whereas matrices are
// found via the ArgObjectR approach.
template<typename R, typename MM, typename MS, typename SM> R WithMatrixScalarCombination (lua_State * L, MM && both, MS && mat_scalar, SM && scalar_mat, int arg1, int arg2)
{
	ArgObjectR<R> o1{L, arg1}, o2{L, arg2};

	if (!o2.mObject) return mat_scalar(*o1.mObject, o2.mScalar);
	else if (!o1.mObject) return scalar_mat(o1.mScalar, *o2.mObject);
	else
	{
		TwoMatrices<R>::CheckTwo(L, o1.mObject != nullptr && o2.mObject != nullptr, arg1, arg2);

		return both(*o1.mObject, *o2.mObject);
	}
}

// Veneer over the LinSpaced factory that also allows for complex types.
template<typename T, int R, int C> struct LinSpacing {
	using V = Eigen::Matrix<typename T::Scalar, R, C>;

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