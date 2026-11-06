//! A JSON document that keeps what `serde_json::Value` would change: key order, duplicate keys and
//! the exact text of numbers and strings. The installer edits other tools' configuration, so anything
//! it does not touch has to mean the same afterwards. Adding `preserve_order` or
//! `arbitrary_precision` to `serde_json` instead would change them for the whole workspace.

use std::fmt;

use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Doc {
    /// A string, number, boolean or null, as written in the source.
    Leaf(String),
    Arr(Vec<Doc>),
    Obj(Vec<(String, Doc)>),
}

impl Doc {
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        // `Value` checks everything, including the nesting limit that keeps recursion below safe.
        // `RawValue` alone only skims nested content.
        serde_json::from_str::<Value>(text)?;
        let raw: Box<RawValue> = serde_json::from_str(text)?;
        Self::from_raw(&raw)
    }

    fn from_raw(raw: &RawValue) -> Result<Self, serde_json::Error> {
        let text = raw.get().trim();
        match text.as_bytes().first() {
            Some(b'{') => {
                let Entries(entries) = serde_json::from_str(text)?;
                let entries = entries
                    .into_iter()
                    .map(|(key, value)| Ok((key, Self::from_raw(&value)?)))
                    .collect::<Result<_, serde_json::Error>>()?;
                Ok(Self::Obj(entries))
            }
            Some(b'[') => {
                let items: Vec<Box<RawValue>> = serde_json::from_str(text)?;
                let items = items
                    .iter()
                    .map(|item| Self::from_raw(item))
                    .collect::<Result<_, _>>()?;
                Ok(Self::Arr(items))
            }
            _ => Ok(Self::Leaf(text.to_owned())),
        }
    }

    /// Pretty-printed with two spaces and a final newline.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        match self {
            Self::Leaf(text) => out.push_str(text),
            Self::Arr(items) if items.is_empty() => out.push_str("[]"),
            Self::Obj(entries) if entries.is_empty() => out.push_str("{}"),
            Self::Arr(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    separator(out, index, depth + 1);
                    item.write(out, depth + 1);
                }
                newline(out, depth);
                out.push(']');
            }
            Self::Obj(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    separator(out, index, depth + 1);
                    out.push_str(&quote(key));
                    out.push_str(": ");
                    value.write(out, depth + 1);
                }
                newline(out, depth);
                out.push('}');
            }
        }
    }

    pub fn to_value(&self) -> Value {
        // A Doc only ever holds valid JSON, so this cannot fail.
        let mut out = String::new();
        self.write(&mut out, 0);
        serde_json::from_str(&out).unwrap_or(Value::Null)
    }

    /// The value JSON parsers would use: with duplicate keys, the last.
    pub fn get(&self, key: &str) -> Option<&Self> {
        let Self::Obj(entries) = self else {
            return None;
        };
        entries.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Self> {
        let Self::Obj(entries) = self else {
            return None;
        };
        entries
            .iter_mut()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Appends the key. Only for a key that is not there.
    pub fn push_entry(&mut self, key: &str, value: Self) {
        if let Self::Obj(entries) = self {
            entries.push((key.to_owned(), value));
        }
    }

    /// Removes every occurrence of the key.
    pub fn remove_key(&mut self, key: &str) {
        if let Self::Obj(entries) = self {
            entries.retain(|(k, _)| k != key);
        }
    }

    pub fn is_empty_container(&self) -> bool {
        match self {
            Self::Obj(entries) => entries.is_empty(),
            Self::Arr(items) => items.is_empty(),
            Self::Leaf(_) => false,
        }
    }
}

fn separator(out: &mut String, index: usize, depth: usize) {
    if index > 0 {
        out.push(',');
    }
    newline(out, depth);
}

fn newline(out: &mut String, depth: usize) {
    out.push('\n');
    out.push_str(&"  ".repeat(depth));
}

fn quote(key: &str) -> String {
    serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_owned())
}

/// Object members in source order.
struct Entries(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for Entries {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EntriesVisitor;

        impl<'de> Visitor<'de> for EntriesVisitor {
            type Value = Entries;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Entries, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Entries(entries))
            }
        }

        deserializer.deserialize_map(EntriesVisitor)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn keeps_order_numbers_and_escapes() {
        let source =
            r#"{"z":1.10,"a":{"y":12345678901234567890123,"b":"\u00e9\n"},"m":[true,null,[],{}]}"#;
        let doc = Doc::parse(source).expect("parse");
        let expected = r#"{
  "z": 1.10,
  "a": {
    "y": 12345678901234567890123,
    "b": "\u00e9\n"
  },
  "m": [
    true,
    null,
    [],
    {}
  ]
}
"#;
        assert_eq!(doc.render(), expected);
    }

    #[test]
    fn rendering_is_stable() {
        let doc = Doc::parse(r#"{"a":[1,{"b":2}],"c":"d"}"#).expect("parse");
        let once = doc.render();
        assert_eq!(Doc::parse(&once).expect("reparse").render(), once);
        assert_eq!(doc.to_value(), json!({"a": [1, {"b": 2}], "c": "d"}));
    }

    #[test]
    fn duplicate_keys_survive_and_the_last_one_counts() {
        let mut doc = Doc::parse(r#"{"a":1,"a":2}"#).expect("parse");
        assert_eq!(doc.get("a"), Some(&Doc::Leaf("2".to_owned())));
        assert!(doc.render().contains("\"a\": 1"));
        doc.remove_key("a");
        assert_eq!(doc, Doc::Obj(vec![]));
    }

    #[test]
    fn keys_with_escapes_are_decoded_for_lookup_and_reencoded() {
        let doc = Doc::parse(r#"{"a\"b":1}"#).expect("parse");
        assert!(doc.get("a\"b").is_some());
        assert_eq!(doc.to_value(), json!({"a\"b": 1}));
    }

    #[test]
    fn rejects_what_is_not_json() {
        for bad in [
            "",
            "   ",
            "{",
            "{\"a\":1,}",
            "// c\n{}",
            "{} trailing",
            "{\"a\":01}",
            "{\"a\":\"\\q\"}",
            "{\"a\":[1 2]}",
        ] {
            assert!(Doc::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_crash() {
        let deep = format!("{}{}", "[".repeat(200_000), "]".repeat(200_000));
        assert!(Doc::parse(&deep).is_err());
    }
}
