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
inline bool WantsBool(lua_State * L, const char * str, int arg = -1)
{
	arg = CoronaLuaNormalize(L, arg);

	lua_pushstring(L, str);	// ..., arg, ..., opt

	bool bWants = lua_equal(L, arg, -1) != 0;

	lua_pop(L, 1);	// ..., arg, ...

	return bWants;
}

//
template<typename T> struct ArgObject {
	T * mMat{nullptr};
	typename T::Scalar mScalar; // complex numbers preclude easy inclusion in union, so don't bother

	template<bool = Eigen::NumTraits<T::Scalar>::IsComplex> typename T::Scalar AsScalar (lua_State * L, int arg)
	{
		return Complex<Eigen::NumTraits<T::Scalar>::Real>(L, arg);
	}

	template<> typename T::Scalar AsScalar<false> (lua_State * L, int arg)
	{
		return LuaXS::GetArg<T::Scalar>(L, arg);
	}

	ArgObject (void) {}

	ArgObject (lua_State * L, int arg)
	{
		if (LuaXS::IsType(L, FullName<T>(), arg)) mMat = LuaXS::UD<T>(L, arg);

		else mScalar = AsScalar(L, arg);
	}
};

//
template<typename T, int R, int C> struct LinSpacing {
	using V = Eigen::Matrix<typename T::Scalar, R, C>;

	template<bool = !Eigen::NumTraits<T::Scalar>::IsComplex> static V Do (lua_State * L, int n)
	{
		return V::LinSpaced(n, ArgObject<T>{}.AsScalar(L, 2), ArgObject<T>{}.AsScalar(L, 3));
	}

	template<> static V Do<false> (lua_State * L, int n)
	{
		using RealV = Eigen::Matrix<T::Scalar::value_type, R, C>;

		T::Scalar low = ArgObject<T>{}.AsScalar(L, 2), high = ArgObject<T>{}.AsScalar(L, 3);
		V cv;

		cv.resize(n);

		cv.real() = RealV::LinSpaced(n, low.real(), high.real());
		cv.imag() = RealV::LinSpaced(n, low.imag(), high.imag());

		return cv;
	}
};

//
template<typename T> struct TwoMatrices {
	T mK, * mMat1, * mMat2;

	TwoMatrices (lua_State * L, int arg1 = 1, int arg2 = 2)
	{
		ArgObject<T> o1{L, arg1}, o2{L, arg2};

		if (o1.mMat && o2.mMat)
		{
			mMat1 = o1.mMat;
			mMat2 = o2.mMat;
		}

		else if (o1.mMat)
		{
			mMat1 = o1.mMat;
			mMat2 = &mK;

			mK.setConstant(mMat1->rows(), mMat1->cols(), o2.mScalar);
		}

		else
		{
			mMat1 = &mK;
			mMat2 = o2.mMat;

			mK.setConstant(mMat2->rows(), mMat2->cols(), o1.mScalar);
		}
	}
};

//
template<typename T, typename R, typename MM, typename MS, typename SM> static R WithMatrixOrScalar (lua_State * L, MM && both, MS && mat_scalar, SM && scalar_mat, int arg1, int arg2)
{
	ArgObject<T> o1{L, arg1};
	ArgObject<R> o2{L, arg2};

	if (!o2.mMat) return mat_scalar(*o1.mMat, o2.mScalar);
	else if (!o1.mMat) return scalar_mat(o1.mScalar, *o2.mMat);
	else return both(*o1.mMat, *o2.mMat);
}