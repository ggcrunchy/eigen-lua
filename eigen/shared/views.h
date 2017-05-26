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
#include "macros.h"
#include "types.h"
#include "utils.h"
#include "views.h"
#include <Eigen/Eigen>
#include <type_traits>
#include <utility>

// TODO: might replace the functions that produce these by some more lightweight packets and collapse
// to matrices otherwise, since they seem to complicate builds with maps and hunting down all the
// scalar ops seems unpleasant
template<typename U, typename V> struct AuxTypeName<Eigen::CwiseUnaryOp<U, V>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "CwiseUnaryOp<");

		if (std::is_same<U, Eigen::internal::scalar_conjugate_op<V::Scalar>>::value)
		{
			luaL_addstring(B, "scalar_conjugate_op<");

			AuxTypeName<V::Scalar>(B, L);

			luaL_addstring(B, ">");
		}

		luaL_addstring(B, ", ");

		AuxTypeName<V>(B, L);

		luaL_addstring(B, ">");
	}
};