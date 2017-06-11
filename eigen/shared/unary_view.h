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

#include "types.h"
#include "views.h"

/*********************
* Unary view methods *
*********************/
template<typename OP, typename V, typename R> struct AttachMethods<Eigen::CwiseUnaryView<OP, V>, R> : InstanceGetters<Eigen::CwiseUnaryView<OP, V>, R> {
	//
	template<typename = OP> struct IsRealOp : std::false_type {};
	template<typename S> struct IsRealOp<Eigen::internal::scalar_real_ref_op<S>> : std::true_type {};

	template<bool = IsLvalue<V>::value> void AddWriteOps (lua_State * L)
	{
		using Real = typename Eigen::NumTraits<typename T::Scalar>::Real;

		luaL_Reg methods[] = {
			"assign", [](lua_State * L)
			{
				auto op = IsRealOp<>::value ? &Eigen::CwiseUnaryView<OP, V>::real : &Eigen::CwiseUnaryView<OP, V>::imag;

				if (HasType<Eigen::CwiseUnaryView<OP, V>>(L, 2)) *GetT(L) = (LuaXS::UD<Eigen::CwiseUnaryView<OP, V>>(L, 2)->*op)();
				else if (HasType<Eigen::CwiseUnaryOp<OP, const V>>(L, 2)) *GetT(L) = (LuaXS::UD<Eigen::CwiseUnaryOp<OP, const V>>(L, 2)->*op)();
				else if (HasType<R>(L, 2)) *GetT(L) = (LuaXS::UD<R>(L, 2)->*op)();
				else
				{																								
					MatrixOf<Real> result;

					*GetT(L) = (SetTemp(L, &result, 2)->*op)();
				}		

				return 0;
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void AddWriteOps<false> (lua_State *) {}

	AttachMethods (lua_State * L)
	{
		AddWriteOps(L);
	}
};