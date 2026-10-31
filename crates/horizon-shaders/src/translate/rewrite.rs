//! The second pass: the stage's code written again for naga, line for line, and the
//! declarations of ours it needs.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use naga::ShaderStage;

use super::scan::{LATER_KEYWORDS, Stage, UNIFORM_STATE, VERTEX_INPUTS, is_ident, significant};
use super::{Sampler, split_sampler};

impl Stage<'_> {
    /// Declarations of ours this stage uses (after the uniform block, samplers and varyings).
    pub(super) fn prelude(&self) -> String {
        let mut p = String::from(
            "float hz_modf(float x, out float i) { i = trunc(x); return x - i; }\n\
             vec2 hz_modf(vec2 x, out vec2 i) { i = trunc(x); return x - i; }\n\
             vec3 hz_modf(vec3 x, out vec3 i) { i = trunc(x); return x - i; }\n\
             vec4 hz_modf(vec4 x, out vec4 i) { i = trunc(x); return x - i; }\n\
             struct hz_FogParameters { vec4 color; float density; float start; float end; float scale; };\n\
             hz_FogParameters hz_Fog;\n",
        );
        if self.gbuffers {
            // Set by our `main` from the frame's matrices (gbuffers.rs).
            p.push_str(
                "mat4 hz_ModelViewMatrix; mat4 hz_ProjectionMatrix; mat4 hz_ModelViewProjectionMatrix;\n\
                 mat4 hz_ModelViewMatrixInverse; mat4 hz_ProjectionMatrixInverse; mat3 hz_NormalMatrix; mat4 hz_TextureMatrix[8];\n",
            );
        } else {
            p.push_str(
                "const mat4 hz_ModelViewMatrix = mat4(1.0); const mat4 hz_ProjectionMatrix = mat4(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0);\n\
                 const mat4 hz_ModelViewProjectionMatrix = hz_ProjectionMatrix; const mat4 hz_ModelViewMatrixInverse = mat4(1.0); const mat3 hz_NormalMatrix = mat3(1.0);\n\
                 const mat4 hz_ProjectionMatrixInverse = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.5, 0.0, 1.0);\n\
                 const mat4 hz_TextureMatrix[8] = mat4[8](mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0), mat4(1.0));\n",
            );
        }
        if self.tokens.contains(&"textureGather") {
            // naga's GLSL frontend has no `textureGather`: the four texels a bilinear sample of
            // `uv` reads, in the order the built-in returns them, component `c` of each.
            p.push_str(
                "vec4 hz_textureGather(texture2D t, sampler s, vec2 uv, int c) { \
                 ivec2 m = textureSize(sampler2D(t, s), 0) - 1; \
                 ivec2 b = ivec2(floor(uv * vec2(m + 1) - 0.5)); \
                 return vec4(texelFetch(sampler2D(t, s), clamp(b + ivec2(0, 1), ivec2(0), m), 0)[c], \
                 texelFetch(sampler2D(t, s), clamp(b + ivec2(1, 1), ivec2(0), m), 0)[c], \
                 texelFetch(sampler2D(t, s), clamp(b + ivec2(1, 0), ivec2(0), m), 0)[c], \
                 texelFetch(sampler2D(t, s), clamp(b, ivec2(0), m), 0)[c]); } \
                 vec4 hz_textureGather(texture2D t, sampler s, vec2 uv) { return hz_textureGather(t, s, uv, 0); }\n",
            );
        }
        if self.kind == ShaderStage::Vertex {
            p.push_str(
                "const vec2 hz_quad[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));\n\
                 vec4 hz_Vertex; vec4 hz_Color; vec4 hz_SecondaryColor; vec3 hz_Normal; float hz_FogCoord;\n\
                 vec4 hz_MultiTexCoord0; vec4 hz_MultiTexCoord1; vec4 hz_MultiTexCoord2; vec4 hz_MultiTexCoord3;\n\
                 vec4 hz_MultiTexCoord4; vec4 hz_MultiTexCoord5; vec4 hz_MultiTexCoord6; vec4 hz_MultiTexCoord7;\n",
            );
        } else {
            for n in &self.outputs {
                if !self.frag_outs.iter().any(|(at, _)| at == n) {
                    writeln!(p, "layout(location = {n}) out vec4 hz_out{n};").unwrap();
                }
            }
            for (_, decl) in &self.frag_outs {
                writeln!(p, "{decl}").unwrap();
            }
            p.push_str("vec4 hz_FragCoord;\n");
        }
        p
    }

    /// The pack's code with the rewrites above; the same number of lines.
    pub(super) fn rewrite(
        &self,
        samplers: &[Sampler],
        uniforms: &BTreeMap<String, String>,
        remainders: bool,
        mixed: bool,
    ) -> String {
        let rem = if remainders {
            self.remainder_calls()
        } else {
            Remainders::default()
        };
        let removed = |i: usize| self.removed.iter().any(|r| r.contains(&i));
        let sampler_ty: HashMap<&str, &str> = samplers
            .iter()
            .map(|s| (s.name.as_str(), s.ty.as_str()))
            .collect();
        let old = self.pp.version.as_ref().is_none_or(|(v, _)| *v < 400);
        let fragment = self.kind == ShaderStage::Fragment;
        let mut out = String::with_capacity(self.pp.code.len() + 1024);
        // Per open bracket: the function called, and the closing text to add.
        let mut calls: Vec<(&str, &str, usize)> = Vec::new();
        // Sampler parameters of the function being written: name → type.
        let mut params: HashMap<String, String> = HashMap::new();
        // Names of samplers a local variable or parameter of the function hides.
        let mut hidden: Vec<&str> = Vec::new();
        // `vec4 tex = texture(tex, uv);`: the variable hides the sampler from the end of its
        // initializer on (GLSL's scope rule): the token there, and the name.
        let mut hide_at: Vec<(usize, &str)> = Vec::new();
        // The function being written: its return type when GLSL converts what it returns
        // (`return 10;` in a `float` function), and whether its last statement so far returns.
        let mut fn_type: Option<&str> = None;
        let (mut statement_start, mut last_returns) = (false, false);
        // Text to write before a token: the `)` of `return float(...)`.
        let mut close_before: HashMap<usize, usize> = HashMap::new();
        // Calls on a shadow sampler whose result is swizzled: `vec4(textureLod(s, p, 0)).x`.
        let mut swizzled: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut depth = 0;
        let mut skip_until = 0;
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let prev = |i: usize| (0..i).rev().find(|&j| significant(self.tokens[j]));
        for (i, &t) in self.tokens.iter().enumerate() {
            hide_at.retain(|&(at, name)| {
                if at == i {
                    hidden.push(name);
                }
                at != i
            });
            if let Some(n) = close_before.remove(&i) {
                out.push_str(&")".repeat(n));
            }
            if depth == 1 && significant(t) && statement_start {
                last_returns = t == "return";
                statement_start = false;
            }
            if let Some(&(close, open)) = rem.before.get(&i) {
                out.push_str(&")".repeat(close));
                out.push_str(&"hz_rem(".repeat(open));
            }
            if i < skip_until {
                continue;
            }
            if rem.percent.contains(&i) {
                out.push(',');
                continue;
            }
            if let Some(text) = self.replaced.get(&i) {
                out.push_str(text);
                continue;
            }
            if removed(i) {
                out.extend(t.chars().filter(|&c| c == '\n'));
                if t.contains(' ') && !t.contains('\n') {
                    out.push(' ');
                }
                continue;
            }
            let called = next(i).is_some_and(|j| self.tokens[j] == "(");
            match t {
                "{" => {
                    if depth == 0 {
                        fn_type = prev(i)
                            .filter(|&j| self.tokens[j] == ")")
                            .and_then(|j| self.function_type(j));
                    }
                    depth += 1;
                    statement_start = depth == 1;
                }
                "}" => {
                    if depth == 1
                        && let Some(ty) = fn_type
                        && !last_returns
                    {
                        // GLSL lets a function end without returning; naga does not.
                        write!(out, "return {ty}(0); ").unwrap();
                    }
                    depth -= 1;
                    if depth == 0 {
                        params.clear();
                        hidden.clear();
                        fn_type = None;
                    }
                    statement_start = depth == 1;
                    last_returns = false;
                }
                ";" => statement_start = depth == 1,
                "(" => {
                    let callee = prev(i).map_or("", |j| self.tokens[j]);
                    let vertex_sample = !fragment && IMPLICIT_LOD.contains(&callee);
                    let close = match callee {
                        _ if swizzled.contains(&i) && vertex_sample => ", 0.0)",
                        _ if swizzled.contains(&i) => ")",
                        "shadow2D" if vertex_sample => ", 0.0)",
                        "shadow2D" | "shadow2DLod" | "shadow2DProj" => ")",
                        // A vertex stage has no derivatives: GLSL samples its level 0.
                        _ if vertex_sample => ", 0.0",
                        _ => "",
                    };
                    calls.push((callee, close, 0));
                }
                ")" => {
                    if let Some((_, close, _)) = calls.pop() {
                        out.push_str(close);
                    }
                }
                "," => {
                    if let Some(c) = calls.last_mut() {
                        c.2 += 1;
                    }
                }
                _ => {}
            }
            // `T x[] = T[](a, b)`: naga gives unsized arrays no size, so the count is written in.
            if t == "["
                && let Some(close) = next(i).filter(|&j| self.tokens[j] == "]")
            {
                if let Some(n) = self.array_length(close) {
                    write!(out, "[{n}]").unwrap();
                    skip_until = close + 1;
                    continue;
                }
            }
            if !is_ident(t) {
                out.push_str(t);
                continue;
            }
            // A function definition's sampler parameters: `sampler2D s` → `texture2D s, sampler hz_s_s`.
            if depth == 0 && t.contains("sampler") && calls.len() == 1 {
                if let Some(n) = next(i).filter(|&n| is_ident(self.tokens[n])) {
                    let name = self.tokens[n];
                    let (texture, sampler) = split_sampler(t);
                    let renamed = keyword_name(name);
                    write!(out, "{texture} {renamed}, {sampler} hz_s_{renamed}").unwrap();
                    params.insert(name.to_owned(), t.to_owned());
                    skip_until = n + 1;
                    continue;
                }
            }
            // `float specular`: a variable of the same name as a sampler, from here to the
            // function's end; `.specular` a member.
            if sampler_ty.contains_key(t) && !params.contains_key(t) {
                let before = prev(i).map(|j| self.tokens[j]);
                if before == Some(".") {
                    out.push_str(t);
                    continue;
                }
                if before.is_some_and(|b| is_ident(b) && !matches!(b, "return" | "case" | "else")) {
                    match next(i).filter(|&j| self.tokens[j] == "=") {
                        Some(eq) => {
                            hide_at.push((self.initializer_end(eq), t));
                            out.push_str(t);
                            continue;
                        }
                        None => hidden.push(t),
                    }
                }
            }
            if hidden.contains(&t) {
                out.push_str(t);
                continue;
            }
            let sampler = params
                .get(t)
                .map(String::as_str)
                .or_else(|| sampler_ty.get(t).copied());
            if let Some(ty) = sampler {
                let as_argument = calls.last().is_some_and(|(callee, _, arg)| {
                    (*callee == "textureGather" && *arg == 0)
                        || self
                            .sampler_functions
                            .get(*callee)
                            .is_some_and(|p| p.get(*arg) == Some(&true))
                }) && prev(i)
                    .is_some_and(|j| matches!(self.tokens[j], "(" | ","));
                let t = keyword_name(t);
                if as_argument {
                    write!(out, "{t}, hz_s_{t}").unwrap();
                } else {
                    write!(out, "{ty}({t}, hz_s_{t})").unwrap();
                }
                continue;
            }
            if uniforms.get(t).is_some_and(|ty| ty == "bool") {
                write!(out, "(hz_bool_{t} != 0)").unwrap();
                continue;
            }
            // `textureLod(shadowtex, p, 0).x`: GLSL swizzles the scalar a shadow sampler gives.
            if called
                && t.starts_with("texture")
                && let Some(open) = next(i)
                && let Some(first) = next(open)
                && params
                    .get(self.tokens[first])
                    .map(String::as_str)
                    .or_else(|| sampler_ty.get(self.tokens[first]).copied())
                    .is_some_and(|ty| ty.contains("Shadow"))
                && let Some(close) = self.matching(open)
                && next(close).is_some_and(|j| self.tokens[j] == ".")
            {
                swizzled.insert(open);
                out.push_str("vec4(");
            }
            let replaced: Option<String> = match t {
                "main" if depth == 0 && called => Some("hz_main".into()),
                "textureGather" if called => Some("hz_textureGather".into()),
                "min" | "max" if called && mixed => Some(format!("hz_{t}")),
                // `return 10;` in a `float` function: GLSL converts, naga does not.
                "return"
                    if depth > 0
                        && fn_type.is_some()
                        && next(i).is_some_and(|j| self.tokens[j] != ";") =>
                {
                    *close_before.entry(self.initializer_end(i)).or_default() += 1;
                    Some(format!("return {}(", fn_type.unwrap_or_default()))
                }
                // `in const float x`: a parameter qualifier GLSL takes and naga's frontend does not.
                "const" if depth == 0 && !calls.is_empty() => Some(String::new()),
                // naga's frontend leaves the code after an unconditional `discard` or `return;`
                // unreachable but half-built (a module that fails validation): behind an `if`
                // it stays reachable as far as naga knows.
                "discard" => Some("if (true) discard".into()),
                "return" if next(i).is_some_and(|j| self.tokens[j] == ";") => {
                    Some("if (true) return".into())
                }
                // GLSL's `modf`, which naga's frontend lacks.
                "modf" if called => Some("hz_modf".into()),
                _ if !fragment && called && IMPLICIT_LOD.contains(&t) => Some(match t {
                    "shadow2D" => "vec4(textureLod".into(),
                    _ => "textureLod".into(),
                }),
                "texture2D" | "texture1D" | "texture3D" | "textureCube" if called => {
                    Some("texture".into())
                }
                "texture2DLod" | "texture3DLod" | "texture2DLodARB" | "textureCubeLod"
                    if called =>
                {
                    Some("textureLod".into())
                }
                "texture2DProj" if called => Some("textureProj".into()),
                "texelFetch2D" | "texelFetch3D" | "texelFetch1D" if called => {
                    Some("texelFetch".into())
                }
                "texture2DGrad" | "texture2DGradARB" if called => Some("textureGrad".into()),
                "shadow2D" if called => Some("vec4(texture".into()),
                "shadow2DLod" if called => Some("vec4(textureLod".into()),
                "shadow2DProj" if called => Some("vec4(textureProj".into()),
                "ftransform" if called => {
                    // `ftransform()` → the product it stands for; its `()` stays as a no-op call... so skip it.
                    let close = next(i).and_then(next);
                    skip_until = close.map_or(i + 1, |c| c + 1);
                    Some("(hz_ModelViewProjectionMatrix * hz_Vertex)".into())
                }
                "gl_FragColor" => Some("hz_out0".into()),
                "gl_FragData" | "gl_TexCoord" => match self.index_after(i) {
                    Some((n, close)) => {
                        skip_until = close + 1;
                        Some(match t {
                            "gl_FragData" => format!("hz_out{n}"),
                            _ => format!("hz_TexCoord{n}"),
                        })
                    }
                    None => None,
                },
                "gl_Color" if fragment => Some("hz_FrontColor".into()),
                "gl_FragCoord" if !fragment => None,
                "gl_FragCoord" => Some("hz_FragCoord".into()),
                "gl_FrontColor" => Some("hz_FrontColor".into()),
                "gl_Fog" => Some("hz_Fog".into()),
                "gl_VertexID" => Some("gl_VertexIndex".into()),
                "gl_InstanceID" => Some("gl_InstanceIndex".into()),
                _ if UNIFORM_STATE.iter().any(|b| t.starts_with(b)) => {
                    Some(format!("hz_{}", &t[3..]))
                }
                _ if !fragment && VERTEX_INPUTS.iter().any(|b| t.starts_with(b)) => {
                    Some(format!("hz_{}", &t[3..]))
                }
                _ if old && LATER_KEYWORDS.contains(&t) => Some(format!("hz_kw_{t}")),
                // OpenGL's GLSL has no `sampler` type; Vulkan's, which naga reads, does.
                "sampler" => Some("hz_kw_sampler".into()),
                _ => None,
            };
            out.push_str(replaced.as_deref().unwrap_or(t));
        }
        out
    }
}

/// Where `a % b` becomes `hz_rem(a, b)`: per token, the `)` to close and the `hz_rem(` to open
/// before it, and the `%` tokens written as `,`.
#[derive(Default)]
pub(super) struct Remainders {
    before: HashMap<usize, (usize, usize)>,
    percent: std::collections::HashSet<usize>,
}

/// `hz_rem` for every pair of operands GLSL's `%` takes: naga's frontend widens a scalar beside a
/// vector for `+`, `-`, `/` and the bitwise operators, but not for `%` (`ivec2(p) % 64`).
pub(super) fn remainder_functions() -> String {
    let mut s = String::new();
    for (scalar, vector) in [("int", "ivec"), ("uint", "uvec"), ("float", "vec")] {
        write!(
            s,
            "{scalar} hz_rem({scalar} a, {scalar} b) {{ return a % b; }} "
        )
        .unwrap();
        for n in 2..=4 {
            let v = format!("{vector}{n}");
            write!(
                s,
                "{v} hz_rem({v} a, {v} b) {{ return a % b; }} \
                 {v} hz_rem({v} a, {scalar} b) {{ return a % {v}(b); }} \
                 {v} hz_rem({scalar} a, {v} b) {{ return {v}(a) % b; }} "
            )
            .unwrap();
        }
    }
    s.push('\n');
    s
}

/// Sampling calls that pick their level from derivatives: in a vertex stage, `textureLod` of 0.
const IMPLICIT_LOD: [&str; 6] = [
    "texture",
    "texture1D",
    "texture2D",
    "texture3D",
    "textureCube",
    "shadow2D",
];

/// A name that is a keyword in the GLSL naga reads, written as ours.
fn keyword_name(name: &str) -> std::borrow::Cow<'_, str> {
    match name {
        "sampler" => "hz_kw_sampler".into(),
        _ => name.into(),
    }
}

/// `hz_min` and `hz_max` for operands of different scalar types: GLSL converts `min(u, 4)` to
/// `uint`, where naga's frontend finds two built-ins as good and refuses.
pub(super) fn mixed_functions() -> String {
    let mut s = String::new();
    let pairs = [
        ("int", "int", "int"),
        ("uint", "uint", "uint"),
        ("float", "float", "float"),
        ("uint", "int", "uint"),
        ("int", "uint", "uint"),
        ("float", "int", "float"),
        ("int", "float", "float"),
        ("float", "uint", "float"),
        ("uint", "float", "float"),
    ];
    for f in ["min", "max"] {
        for (a, b, r) in pairs {
            write!(
                s,
                "{r} hz_{f}({a} a, {b} b) {{ return {f}({r}(a), {r}(b)); }} "
            )
            .unwrap();
        }
        for (scalar, vector) in [("int", "ivec"), ("uint", "uvec"), ("float", "vec")] {
            for n in 2..=4 {
                let v = format!("{vector}{n}");
                write!(
                    s,
                    "{v} hz_{f}({v} a, {v} b) {{ return {f}(a, b); }} \
                     {v} hz_{f}({v} a, {scalar} b) {{ return {f}(a, {v}(b)); }} "
                )
                .unwrap();
            }
        }
    }
    s.push('\n');
    s
}

/// Words that take a parenthesis without being called.
const NOT_CALLS: [&str; 8] = [
    "if", "while", "for", "switch", "return", "else", "case", "do",
];

impl Stage<'_> {
    /// Every `a % b` in a function's body, as `hz_rem(a, b)`: the left operand is the run of
    /// multiplicative terms before the `%` (`%` binds as `*` and `/`, left to right), the right
    /// one the unary expression after it. A `%` whose operands are not plain is left as it is.
    pub(super) fn remainder_calls(&self) -> Remainders {
        let sig: Vec<usize> = (0..self.tokens.len())
            .filter(|&i| significant(self.tokens[i]))
            .collect();
        let tok = |k: usize| self.tokens[sig[k]];
        let n = sig.len();
        let mut partner = vec![usize::MAX; n];
        let mut stack = Vec::new();
        for k in 0..n {
            match tok(k) {
                "(" | "[" | "{" => stack.push(k),
                ")" | "]" | "}" => {
                    if let Some(o) = stack.pop() {
                        partner[o] = k;
                        partner[k] = o;
                    }
                }
                _ => {}
            }
        }
        let number = |t: &str| t.starts_with(|c: char| c.is_ascii_digit() || c == '.') && t != ".";
        let call = |k: usize| k > 0 && is_ident(tok(k - 1)) && !NOT_CALLS.contains(&tok(k - 1));
        // The first token of the postfix expression ending at `k`.
        let term_start = |mut k: usize| -> Option<usize> {
            loop {
                match tok(k) {
                    ")" => {
                        let o = *partner.get(k).filter(|&&o| o != usize::MAX)?;
                        k = if call(o) { o - 1 } else { o };
                    }
                    "]" => {
                        let o = *partner.get(k).filter(|&&o| o != usize::MAX)?;
                        k = o.checked_sub(1)?;
                        continue;
                    }
                    t if (is_ident(t) && !NOT_CALLS.contains(&t)) || number(t) => {}
                    _ => return None,
                }
                if k >= 2 && tok(k - 1) == "." {
                    k -= 2;
                    continue;
                }
                return Some(k);
            }
        };
        // The last token of the unary expression starting at `k`.
        let term_end = |mut k: usize| -> Option<usize> {
            while k < n && matches!(tok(k), "-" | "+" | "!" | "~") {
                k += 1;
            }
            let t = *self.tokens.get(*sig.get(k)?)?;
            if t == "(" {
                k = *partner.get(k).filter(|&&o| o != usize::MAX)?;
            } else if is_ident(t) && !NOT_CALLS.contains(&t) {
                if k + 1 < n && tok(k + 1) == "(" {
                    k = *partner.get(k + 1).filter(|&&o| o != usize::MAX)?;
                }
            } else if !number(t) {
                return None;
            }
            loop {
                if k + 1 < n && tok(k + 1) == "[" {
                    k = *partner.get(k + 1).filter(|&&o| o != usize::MAX)?;
                } else if k + 2 < n && tok(k + 1) == "." && is_ident(tok(k + 2)) {
                    k += 2;
                    if k + 1 < n && tok(k + 1) == "(" {
                        k = *partner.get(k + 1).filter(|&&o| o != usize::MAX)?;
                    }
                } else {
                    return Some(k);
                }
            }
        };
        let mut r = Remainders::default();
        let mut depth = 0;
        for k in 0..n {
            match tok(k) {
                "{" => depth += 1,
                "}" => depth -= 1,
                "%" if depth > 0 && k > 0 && k + 1 < n && tok(k + 1) != "=" => {
                    // The left operand: terms joined by `*`, `/` and `%`.
                    let Some(mut start) = term_start(k - 1) else {
                        continue;
                    };
                    while start >= 2 && matches!(tok(start - 1), "*" | "/" | "%") {
                        match term_start(start - 2) {
                            Some(s) => start = s,
                            None => break,
                        }
                    }
                    let Some(end) = term_end(k + 1) else {
                        continue;
                    };
                    r.before.entry(sig[start]).or_default().1 += 1;
                    r.before.entry(sig[end] + 1).or_default().0 += 1;
                    r.percent.insert(sig[k]);
                }
                _ => {}
            }
        }
        r
    }

    /// The `)` that closes the bracket at `open`.
    pub(super) fn matching(&self, open: usize) -> Option<usize> {
        let mut nesting = 0;
        for (j, &t) in self.tokens.iter().enumerate().skip(open) {
            match t {
                "(" | "[" | "{" => nesting += 1,
                ")" | "]" | "}" => {
                    nesting -= 1;
                    if nesting == 0 {
                        return Some(j);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The return type of the function whose parameter list ends at `close`, when it is one
    /// GLSL converts to (a scalar, vector or matrix); `None` for `void`, structs and arrays.
    pub(super) fn function_type(&self, close: usize) -> Option<&str> {
        let mut nesting = 0;
        let mut open = None;
        for j in (0..=close).rev() {
            match self.tokens[j] {
                ")" => nesting += 1,
                "(" => {
                    nesting -= 1;
                    if nesting == 0 {
                        open = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        let significant_before = |i: usize| (0..i).rev().find(|&j| significant(self.tokens[j]));
        let name = significant_before(open?)?;
        if !is_ident(self.tokens[name]) || NOT_CALLS.contains(&self.tokens[name]) {
            return None;
        }
        let ty = self.tokens[significant_before(name)?];
        let base = ty.trim_start_matches(['i', 'u', 'b', 'd']);
        let convertible = matches!(ty, "float" | "int" | "uint" | "bool" | "double")
            || base.starts_with("vec")
            || ty.starts_with("mat");
        convertible.then_some(ty)
    }

    /// The `,` or `;` that ends the initializer whose `=` is at `eq`.
    pub(super) fn initializer_end(&self, eq: usize) -> usize {
        let mut nesting = 0;
        for (j, &t) in self.tokens.iter().enumerate().skip(eq + 1) {
            match t {
                "(" | "[" | "{" => nesting += 1,
                ")" | "]" | "}" => nesting -= 1,
                "," | ";" if nesting <= 0 => return j,
                _ => {}
            }
        }
        self.tokens.len()
    }

    /// The element count of an unsized array whose `[]` ends at `close`: the arguments of the
    /// constructor that follows (`[] = T[](...)` or `[](...)`).
    pub(super) fn array_length(&self, close: usize) -> Option<usize> {
        let mut j = close + 1;
        let mut seen_eq = false;
        while let Some(&t) = self.tokens.get(j) {
            match t {
                "(" => break,
                "=" if !seen_eq => seen_eq = true,
                ";" | "," | ")" | "{" => return None,
                _ => {}
            }
            j += 1;
        }
        let (mut depth, mut args, mut any) = (0, 1, false);
        for &t in &self.tokens[j + 1..] {
            match t {
                "(" | "[" | "{" => depth += 1,
                ")" if depth == 0 => return any.then_some(args),
                ")" | "]" | "}" => depth -= 1,
                "," if depth == 0 => args += 1,
                _ if significant(t) => any = true,
                _ => {}
            }
        }
        None
    }
}

/// What our vertex `main` gives a vertex input of a composite-style program.
pub(super) fn attribute_value(name: &str, ty: &str) -> String {
    match name {
        "vaPosition" => "vec3(hz_p, 0.0)".into(),
        "vaUV0" => "hz_p".into(),
        "vaColor" => "vec4(1.0)".into(),
        "vaNormal" => "vec3(0.0, 0.0, 1.0)".into(),
        _ => format!("{ty}(0)"),
    }
}
