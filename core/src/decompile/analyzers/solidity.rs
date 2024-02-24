use core::panic;
use std::str;

use ethers::{
    abi::{decode, AbiEncode, ParamType},
    prelude::U256,
};
use heimdall_common::{
    ether::evm::{
        core::{
            opcodes::WrappedOpcode,
            types::{byte_size_to_type, convert_bitmask},
            vm::{Instruction, State},
        },
        ext::exec::VMTrace,
    },
    utils::{
        io::logging::TraceFactory,
        strings::{decode_hex, encode_hex_reduced},
    },
};

use super::super::{constants::AND_BITMASK_REGEX, precompile::decode_precompile};
use crate::{
    decompile::{
        constants::VARIABLE_SIZE_CHECK_REGEX,
        util::{CalldataFrame, Function, StorageFrame},
    },
    error::Error,
};

/// Converts a VMTrace to a Function through lexical and syntactic analysis
///
/// ## Parameters
/// - `vm_trace` - The VMTrace to be analyzed
/// - `function` - The function to be updated with the analysis results
/// - `trace` - The TraceFactory to be updated with the analysis results
/// - `trace_parent` - The parent of the current VMTrace
/// - `conditional_map` - A map of the conditionals in the current trace
/// - `branch` - Branch metadata for the current trace. In the format of (branch_depth,
///   branch_index)
///     - @jon-becker: This will be used later to determin if a condition is a require
///
///
/// ## Returns
/// - `function` - The function updated with the analysis results
// TODO: `analyze_sol` is too long and needs to be refactored into a series of smaller functions.
// this will improve readability as well as test coverage
pub fn analyze_sol(
    vm_trace: &VMTrace,
    function: Function, // func 1 => til the end of this execution
    trace: &mut TraceFactory,
    trace_parent: u32,
    conditional_map: &mut Vec<String>,
    branch: (u32, u8),
) -> Result<Function, Error> {
    // make a clone of the recursed analysis function
    let mut function_mut = function;
    let mut jumped_conditional: Option<String> = None;

    // perform analysis on the operations of the current VMTrace branch
    for operation in &vm_trace.operations {
        // TODO: where there's clones, need to be refactored to use references
        // because otherwise the next iteration will not have the context of the previous one
        function = FunctionAnalyzer::new(
            operation.clone(),
            &function,
            trace.clone(),
            trace_parent,
            conditional_map.clone(),
            jumped_conditional.clone(),
        )
        .analyze()
        .unwrap();
    }

    // recurse into the children of the VMTrace map
    for (i, child) in vm_trace.children.iter().enumerate() {
        function = analyze_sol(
            child,
            function,
            trace,
            trace_parent,
            conditional_map,
            (branch.0 + 1, i as u8),
        )?;
    }

    // check if the ending brackets are needed
    if jumped_conditional.is_some()
        && conditional_map.contains(
            &jumped_conditional
                .clone()
                .expect("impossible case: should have short-circuited in previous conditional"),
        )
    {
        // remove the conditional
        for (i, conditional) in conditional_map.iter().enumerate() {
            if conditional
                == &jumped_conditional
                    .clone()
                    .expect("impossible case: should have short-circuited in previous conditional")
            {
                conditional_map.remove(i);
                break;
            }
        }

        function.logic.push("}".to_string());
    }

    Ok(function)
}

// TODO: consider using lifetime for Function and TraceFactory
// TODO: check for all the breaks, continues and unwraps
/// Given an opcode, return a Function with the opcode's logic added, visibility
/// and purity updated, and storage and memory updated
#[derive(Debug)]
struct FunctionAnalyzer<'a> {
    operation: State,
    function: &'a mut Function,
    trace: TraceFactory,
    trace_parent: u32,
    conditional_map: Vec<String>,
    jumped_conditional: Option<String>,
    instruction: Instruction,
    opcode_name: String,
}

impl<'a> FunctionAnalyzer<'a> {
    pub fn new(
        operation: State,
        function: &'a mut Function,
        trace: TraceFactory,
        trace_parent: u32,
        conditional_map: Vec<String>,
        jumped_conditional: Option<String>,
    ) -> Self {
        Self {
            operation: operation.clone(),
            function,
            trace,
            trace_parent,
            conditional_map,
            jumped_conditional,
            instruction: operation.last_instruction.clone(),
            opcode_name: operation
                .last_instruction
                .opcode_details
                .clone()
                .ok_or(Error::Generic("failed to get opcode details for instruction".to_string()))
                .unwrap() // TODO: shouldn't unwrap
                .name
                .to_string(),
        }
    }

    pub fn analyze(&mut self) -> Result<Function, Error> {
        let _storage = self.operation.storage.clone();
        let opcode_number = self.instruction.opcode;

        // TODO: should be refactored
        let opcode_name_str = self.opcode_name.clone();
        let opcode_name_str = opcode_name_str.as_str();

        if self.function.pure && NON_PURE_OPCODES.contains(&opcode_name_str) {
            self.set_not_pure();
        }

        if self.function.view && NON_VIEW_OPCODES.contains(&opcode_name_str) {
            self.set_not_view();
        }

        if LOG_OPCODES.contains(&opcode_number) {
            self.handle_log();
        } else {
            match self.opcode_name.as_str() {
                "JUMPI" => self.handle_jumpi(),
                "REVERT" => self.handle_revert(),
                "RETURN" => self.handle_return(),
                "SELDFESTRUCT" => self.handle_selfdestruct(),
                "SSTORE" => self.handle_sstore(),
                "MSTORE" => self.handle_mstore(),
                "STATICCALL" => self.handle_staticcall(),
                "DELEGATECALL" => self.handle_delegatecall(),
                "CALL" | "CALLCODE" => self.handle_call_callcode(),
                "CODECOPY" => self.handle_codecopy(),
                "EXTCODECOPY" => self.handle_extcodecopy(),
                "CALLDATACOPY" => self.handle_calldatacopy(),
                "CREATE" => self.handle_create(),
                "CREATE2" => self.handle_create2(),
                "CALLDATALOAD" => self.handle_calldataload(),
                "ISZERO" => self.handle_iszero(),
                "AND" | "OR" => self.handle_and_or(),
                _ if MATH_OPCODES.contains(&opcode_name_str) => self.handle_math(),
                _ if BITWISE_OPCODES.contains(&opcode_name_str) => self.handle_bitwise(),
                // TODO: shouldn't panic
                _ => panic!("No opcode handler found for opcode {}", opcode_name_str),
            }
        }

        Ok(self.function.clone())
    }

    fn set_not_pure(&mut self) {
        self.function.pure = false;
        self.trace.add_info(
            self.trace_parent,
            self.instruction.instruction.try_into().unwrap_or(u32::MAX),
            &format!(
                "instruction {} ({}) indicates an non-pure function.",
                self.instruction.instruction, self.opcode_name
            ),
        );
    }

    fn set_not_view(&mut self) {
        self.function.view = false;
        self.trace.add_info(
            self.trace_parent,
            self.instruction.instruction.try_into().unwrap_or(u32::MAX),
            &format!(
                "instruction {} ({}) indicates a non-view function.",
                self.instruction.instruction, self.opcode_name
            ),
        );
    }

    fn handle_log(&mut self) {
        // LOG0, LOG1, LOG2, LOG3, LOG4
        let logged_event = match self.operation.events.last() {
            Some(event) => event,
            None => {
                self.function.notices.push(format!(
                    "unable to decode event emission at instruction {}",
                    self.instruction.instruction
                ));

                // TODO: shouldn't panic
                panic!(
                    "unable to decode event emission at instruction {}",
                    self.instruction.instruction
                );
            }
        };

        let event_selector = logged_event.topics.first().unwrap_or(&U256::zero()).to_owned();
        let anonymous = event_selector == U256::zero();

        // check to see if the event is a duplicate
        if !self.function.events.iter().any(|(selector, _)| selector == &event_selector) {
            // add the event to the function
            self.function.events.insert(event_selector, (None, logged_event.clone()));

            // decode the data field
            let data_mem_ops = self
                .function
                .get_memory_range(self.instruction.inputs[0], self.instruction.inputs[1]);
            let data_mem_ops_solidified = data_mem_ops
                .iter()
                .map(|x| x.operations.solidify())
                .collect::<Vec<String>>()
                .join(", ");

            let event_name = &logged_event
                .topics
                .first()
                .unwrap_or(&U256::from(0))
                .encode_hex()
                .replacen("0x", "", 1)[0..8];

            let event_topics = match logged_event.topics.get(1..) {
                Some(topics) => match !logged_event.data.is_empty() && !topics.is_empty() {
                    true => {
                        let mut solidified_topics: Vec<String> = Vec::new();
                        for (i, _) in topics.iter().enumerate() {
                            solidified_topics
                                .push(self.instruction.input_operations[i + 3].solidify());
                        }
                        format!("{}, ", solidified_topics.join(", "))
                    }
                    false => {
                        let mut solidified_topics: Vec<String> = Vec::new();
                        for (i, _) in topics.iter().enumerate() {
                            solidified_topics
                                .push(self.instruction.input_operations[i + 3].solidify());
                        }
                        solidified_topics.join(", ")
                    }
                },
                None => "".to_string(),
            };

            // add the event emission to the function's logic
            // will be decoded during post-processing
            self.function.logic.push(format!(
                "emit Event_{}({}{});{}",
                event_name,
                event_topics,
                data_mem_ops_solidified,
                if anonymous { " // anonymous event" } else { "" }
            ));
        }
    }

    fn handle_jumpi(&mut self) {
        // // this is an if conditional for the children branches
        let conditional = self.instruction.input_operations[1].solidify();

        // remove non-payable check and mark function as non-payable
        if conditional == "!msg.value" {
            // this is marking the start of a non-payable function
            self.trace.add_info(
                self.trace_parent,
                self.instruction.instruction.try_into().unwrap_or(u32::MAX),
                &format!(
                    "conditional at instruction {} indicates an non-payble function.",
                    self.instruction.instruction
                ),
            );
            self.function.payable = false;
        }

        // perform a series of checks to determine if the condition
        // is added by the compiler and can be ignored
        if (conditional.contains("msg.data.length") && conditional.contains("0x04"))
            || VARIABLE_SIZE_CHECK_REGEX.is_match(&conditional).unwrap_or(false)
            || (conditional.replace('!', "") == "success")
        {
            ()
        }

        self.function.logic.push(format!("if ({conditional}) {{").to_string());

        // save a copy of the conditional and add it to the conditional map
        self.jumped_conditional = Some(conditional.clone());
        self.conditional_map.push(conditional);
    }

    // TODO: the logic to find the latest if statement is similar on error string
    // and custom error, should be refactored
    fn handle_revert(&mut self) {
        // Safely convert U256 to usize
        let offset: usize = self.instruction.inputs[0].try_into().unwrap_or(0);
        let size: usize = self.instruction.inputs[1].try_into().unwrap_or(0);
        let revert_data = self.operation.memory.read(offset, size);

        // (1) if revert_data starts with 0x08c379a0, the folling is an error string
        // abiencoded (2) if revert_data starts with 0x4e487b71, the
        // following is a compiler panic (3) if revert_data starts with any
        // other 4byte selector, it is a custom error and should
        //     be resolved and added to the generated ABI
        // (4) if revert_data is empty, it is an empty revert. Ex:
        //       - if (true != false) { revert() };
        //       - require(true != false)

        // handle case with error string abiencoded
        if revert_data.starts_with(&[0x08, 0xc3, 0x79, 0xa0]) {
            let revert_string = match revert_data.get(4..) {
                Some(hex_data) => match decode(&[ParamType::String], hex_data) {
                    Ok(revert) => revert[0].to_string(),
                    Err(_) => "decoding error".to_string(),
                },
                None => "decoding error".to_string(),
            };
            match self.jumped_conditional.clone() {
                Some(condition) => {
                    self.function.logic.push(format!("require({condition}, \"{revert_string}\");"));
                }
                None => {
                    // loop backwards through logic to find the last IF statement
                    for i in (0..self.function.logic.len()).rev() {
                        if self.function.logic[i].starts_with("if") {
                            let conditional = match self.conditional_map.pop() {
                                Some(condition) => condition,
                                None => break,
                            };

                            self.function.logic[i] =
                                format!("require({conditional}, \"{revert_string}\");");
                        }
                    }
                }
            }
        }
        // handle case with panics
        else if revert_data.starts_with(&[0x4e, 0x48, 0x7b, 0x71]) {
            ()
        }
        // handle case with custom error OR empty revert
        else {
            let custom_error_placeholder = match revert_data.get(0..4) {
                Some(selector) => {
                    self.function.errors.insert(U256::from(selector), None);
                    format!(
                        "CustomError_{}()",
                        encode_hex_reduced(U256::from(selector)).replacen("0x", "", 1)
                    )
                }
                None => "()".to_string(),
            };

            match self.jumped_conditional.clone() {
                Some(condition) => {
                    let revert_logic = match custom_error_placeholder.as_str() {
                        "()" => format!("require({condition});",),
                        _ => format!("require({condition}, {custom_error_placeholder});"),
                    };

                    self.function.logic.push(revert_logic);
                }
                None => {
                    // loop backwards through logic to find the last IF statement
                    for i in (0..self.function.logic.len()).rev() {
                        if self.function.logic[i].starts_with("if") {
                            let conditional = match self.conditional_map.pop() {
                                Some(condition) => condition,
                                None => break,
                            };

                            self.function.logic[i] = match custom_error_placeholder.as_str() {
                                "()" => format!("require({conditional});",),
                                _ => format!("require({conditional}, {custom_error_placeholder});"),
                            };
                        }
                    }
                }
            }
        }
    }

    fn handle_return(&mut self) {
        // Safely convert U256 to usize
        let size: usize = self.instruction.inputs[1].try_into().unwrap_or(0);

        let return_memory_operations =
            self.function.get_memory_range(self.instruction.inputs[0], self.instruction.inputs[1]);
        let return_memory_operations_solidified = return_memory_operations
            .iter()
            .map(|x| x.operations.solidify())
            .collect::<Vec<String>>()
            .join(", ");

        // we don't want to overwrite the return value if it's already been set
        if self.function.returns == Some(String::from("uint256")) || self.function.returns.is_none()
        {
            // if the return operation == ISZERO, this is a boolean return
            if return_memory_operations.len() == 1
                && return_memory_operations[0].operations.opcode.name == "ISZERO"
            {
                self.function.returns = Some(String::from("bool"));
            } else {
                self.function.returns = match size > 32 {
                    // if the return data is > 32 bytes, we append "memory" to the return
                    // type
                    true => Some(format!("{} memory", "bytes")),
                    false => {
                        // attempt to find a return type within the return memory operations
                        let byte_size = match AND_BITMASK_REGEX
                            .find(&return_memory_operations_solidified)
                            .ok()
                            .flatten()
                        {
                            Some(bitmask) => {
                                let cast = bitmask.as_str();

                                cast.matches("ff").count()
                            }
                            None => 32,
                        };

                        // convert the cast size to a string
                        let (_, cast_types) = byte_size_to_type(byte_size);
                        Some(cast_types[0].to_string())
                    }
                };
            }
        }
        if return_memory_operations.len() <= 1 {
            self.function.logic.push(format!("return {return_memory_operations_solidified};"));
        } else {
            self.function
                .logic
                .push(format!("return abi.encodePacked({return_memory_operations_solidified});"));
        }
    }

    fn handle_selfdestruct(&mut self) {
        let addr = match decode_hex(&self.instruction.inputs[0].encode_hex()) {
            Ok(hex_data) => match decode(&[ParamType::Address], &hex_data) {
                Ok(addr) => addr[0].to_string(),
                Err(_) => "decoding error".to_string(),
            },
            _ => "".to_string(),
        };

        self.function.logic.push(format!("selfdestruct({addr});"));
    }

    fn handle_sstore(&mut self) {
        let key = self.instruction.inputs[0];
        let value = self.instruction.inputs[1];
        let operations = self.instruction.input_operations[1].clone();

        // add the sstore to the function's storage map
        self.function.storage.insert(key, StorageFrame { value, operations });
        self.function.logic.push(format!(
            "storage[{}] = {};",
            self.instruction.input_operations[0].solidify(),
            self.instruction.input_operations[1].solidify(),
        ));
    }

    fn handle_mstore(&mut self) {
        let key = self.instruction.inputs[0];
        let value = self.instruction.inputs[1];
        let operation = self.instruction.input_operations[1].clone();

        // add the mstore to the function's memory map
        self.function.memory.insert(key, StorageFrame { value, operations: operation });
        self.function.logic.push(format!(
            "memory[{}] = {};",
            encode_hex_reduced(key),
            self.instruction.input_operations[1].solidify()
        ));
    }

    fn handle_calldatacopy(&mut self) {
        let memory_offset = &self.instruction.input_operations[0];
        let source_offset = self.instruction.inputs[1];
        let size_bytes = self.instruction.inputs[2];

        // add the mstore to the function's memory map
        self.function.logic.push(format!(
            "memory[{}] = msg.data[{}:{}];",
            memory_offset.solidify(),
            source_offset,
            source_offset.saturating_add(size_bytes)
        ));
    }

    fn handle_codecopy(&mut self) {
        let memory_offset = &self.instruction.input_operations[0];
        let source_offset = self.instruction.inputs[1];
        let size_bytes = self.instruction.inputs[2];

        // add the mstore to the function's memory map
        self.function.logic.push(format!(
            "memory[{}] = this.code[{}:{}]",
            memory_offset.solidify(),
            source_offset,
            source_offset.saturating_add(size_bytes)
        ));
    }

    fn handle_extcodecopy(&mut self) {
        let address = &self.instruction.input_operations[0];
        let memory_offset = &self.instruction.input_operations[1];
        let source_offset = self.instruction.inputs[2];
        let size_bytes = self.instruction.inputs[3];

        // add the mstore to the function's memory map
        self.function.logic.push(format!(
            "memory[{}] = address({}).code[{}:{}]",
            memory_offset.solidify(),
            address.solidify(),
            source_offset,
            source_offset.saturating_add(size_bytes)
        ));
    }

    fn handle_staticcall(&mut self) {
        // if the gas param WrappedOpcode is not GAS(), add the gas param to the function's
        // logic
        let modifier =
            match self.instruction.input_operations[0] != WrappedOpcode::new(0x5A, vec![]) {
                true => format!("{{ gas: {} }}", self.instruction.input_operations[0].solidify()),
                false => String::from(""),
            };

        let address = &self.instruction.input_operations[1];
        let extcalldata_memory =
            self.function.get_memory_range(self.instruction.inputs[2], self.instruction.inputs[3]);

        // check if the external call is a precompiled contract
        match decode_precompile(
            self.instruction.inputs[1],
            extcalldata_memory.clone(),
            self.instruction.input_operations[2].clone(),
        ) {
            (true, precompile_logic) => {
                self.function.logic.push(precompile_logic);
            }
            _ => {
                self.function.logic.push(format!(
                    "(bool success, bytes memory ret0) = address({}).staticcall{}(abi.encode({}));",
                    address.solidify(),
                    modifier,
                    extcalldata_memory
                        .iter()
                        .map(|x| x.operations.solidify())
                        .collect::<Vec<String>>()
                        .join(", "),
                ));
            }
        }
    }

    fn handle_delegatecall(&mut self) {
        // if the gas param WrappedOpcode is not GAS(), add the gas param to the function's
        // logic
        let modifier =
            match self.instruction.input_operations[0] != WrappedOpcode::new(0x5A, vec![]) {
                true => format!("{{ gas: {} }}", self.instruction.input_operations[0].solidify()),
                false => String::from(""),
            };

        let address = &self.instruction.input_operations[1];
        let extcalldata_memory =
            self.function.get_memory_range(self.instruction.inputs[2], self.instruction.inputs[3]);

        // check if the external call is a precompiled contract
        match decode_precompile(
            self.instruction.inputs[1],
            extcalldata_memory.clone(),
            self.instruction.input_operations[2].clone(),
        ) {
            (true, precompile_logic) => {
                self.function.logic.push(precompile_logic);
            }
            _ => {
                self.function.logic.push(format!(
                    "(bool success, bytes memory ret0) = address({}).delegatecall{}(abi.encode({}));",
                    address.solidify(),
                    modifier,
                    extcalldata_memory
                        .iter()
                        .map(|x| x.operations.solidify())
                        .collect::<Vec<String>>()
                        .join(", "),
                ));
            }
        }
    }

    fn handle_call_callcode(&mut self) {
        // if the gas param WrappedOpcode is not GAS(), add the gas param to the function's
        // logic
        let gas = match self.instruction.input_operations[0] != WrappedOpcode::new(0x5A, vec![]) {
            true => format!("gas: {}, ", self.instruction.input_operations[0].solidify()),
            false => String::from(""),
        };
        let value = match self.instruction.input_operations[2] != WrappedOpcode::new(0x5A, vec![]) {
            true => format!("value: {}", self.instruction.input_operations[2].solidify()),
            false => String::from(""),
        };
        let modifier = match !gas.is_empty() || !value.is_empty() {
            true => format!("{{ {gas}{value} }}"),
            false => String::from(""),
        };

        let address = &self.instruction.input_operations[1];
        let extcalldata_memory =
            self.function.get_memory_range(self.instruction.inputs[3], self.instruction.inputs[4]);

        // check if the external call is a precompiled contract
        match decode_precompile(
            self.instruction.inputs[1],
            extcalldata_memory.clone(),
            self.instruction.input_operations[5].clone(),
        ) {
            (is_precompile, precompile_logic) if is_precompile => {
                self.function.logic.push(precompile_logic);
            }
            _ => {
                self.function.logic.push(format!(
                    "(bool success, bytes memory ret0) = address({}).call{}(abi.encode({}));",
                    address.solidify(),
                    modifier,
                    extcalldata_memory
                        .iter()
                        .map(|x| x.operations.solidify())
                        .collect::<Vec<String>>()
                        .join(", ")
                ));
            }
        }
    }

    fn handle_create(&mut self) {
        self.function.logic.push(format!(
            "assembly {{ addr := create({}, {}, {}) }}",
            self.instruction.input_operations[0].solidify(),
            self.instruction.input_operations[1].solidify(),
            self.instruction.input_operations[2].solidify(),
        ));
    }

    fn handle_create2(&mut self) {
        self.function.logic.push(format!(
            "assembly {{ addr := create({}, {}, {}, {}) }}",
            self.instruction.input_operations[0].solidify(),
            self.instruction.input_operations[1].solidify(),
            self.instruction.input_operations[2].solidify(),
            self.instruction.input_operations[3].solidify(),
        ));
    }

    fn handle_calldataload(&mut self) {
        let slot_as_usize: usize = self.instruction.inputs[0].try_into().unwrap_or(usize::MAX);
        let calldata_slot = (slot_as_usize.saturating_sub(4)) / 32;
        match self.function.arguments.get(&calldata_slot) {
            Some(_) => {}
            None => {
                self.function.arguments.insert(
                    calldata_slot,
                    (
                        CalldataFrame {
                            slot: calldata_slot,
                            operation: self.instruction.input_operations[0].to_string(),
                            mask_size: 32,
                            heuristics: Vec::new(),
                        },
                        vec![
                            "bytes".to_string(),
                            "uint256".to_string(),
                            "int256".to_string(),
                            "string".to_string(),
                            "bytes32".to_string(),
                            "uint".to_string(),
                            "int".to_string(),
                        ],
                    ),
                );
            }
        }
    }

    fn handle_iszero(&mut self) {
        if let Some(calldata_slot_operation) = self
            .instruction
            .input_operations
            .iter()
            .find(|operation| operation.opcode.name == "CALLDATALOAD")
        {
            if let Some((calldata_slot, arg)) =
                self.function.arguments.clone().iter().find(|(_, (frame, _))| {
                    frame.operation == calldata_slot_operation.inputs[0].to_string()
                })
            {
                // copy the current potential types to a new vector and remove duplicates
                let mut potential_types = vec![
                    "bool".to_string(),
                    "bytes1".to_string(),
                    "uint8".to_string(),
                    "int8".to_string(),
                ];
                potential_types.append(&mut arg.1.clone());
                potential_types.sort();
                potential_types.dedup();

                // replace mask size and potential types
                self.function.arguments.insert(*calldata_slot, (arg.0.clone(), potential_types));
            }
        };
    }

    fn handle_and_or(&mut self) {
        if let Some(calldata_slot_operation) =
            self.instruction.input_operations.iter().find(|operation| {
                operation.opcode.name == "CALLDATALOAD" || operation.opcode.name == "CALLDATACOPY"
            })
        {
            // convert the bitmask to it's potential solidity types
            let (mask_size_bytes, mut potential_types) = convert_bitmask(self.instruction.clone());

            if let Some((calldata_slot, arg)) =
                self.function.arguments.clone().iter().find(|(_, (frame, _))| {
                    frame.operation == calldata_slot_operation.inputs[0].to_string()
                })
            {
                // append the current potential types to the new vector and remove
                // duplicates
                potential_types.append(&mut arg.1.clone());
                potential_types.sort();
                potential_types.dedup();

                // replace mask size and potential types
                self.function.arguments.insert(
                    *calldata_slot,
                    (
                        CalldataFrame {
                            slot: arg.0.slot,
                            operation: arg.0.operation.clone(),
                            mask_size: mask_size_bytes,
                            heuristics: Vec::new(),
                        },
                        potential_types,
                    ),
                );
            }
        };
    }

    fn handle_math(&mut self) {
        // get the calldata slot operation
        if let Some((key, (frame, potential_types))) =
            self.function.arguments.clone().iter().find(|(_, (frame, _))| {
                self.instruction.output_operations.iter().any(|operation| {
                    operation.to_string().contains(frame.operation.as_str())
                        && !frame.heuristics.contains(&"integer".to_string())
                })
            })
        {
            self.function.arguments.insert(
                *key,
                (
                    CalldataFrame {
                        slot: frame.slot,
                        operation: frame.operation.clone(),
                        mask_size: frame.mask_size,
                        heuristics: vec!["integer".to_string()],
                    },
                    potential_types.to_owned(),
                ),
            );
        }
    }

    fn handle_bitwise(&mut self) {
        // get the calldata slot operation
        if let Some((key, (frame, potential_types))) =
            self.function.arguments.clone().iter().find(|(_, (frame, _))| {
                self.instruction.output_operations.iter().any(|operation| {
                    operation.to_string().contains(frame.operation.as_str())
                        && !frame.heuristics.contains(&"bytes".to_string())
                })
            })
        {
            self.function.arguments.insert(
                *key,
                (
                    CalldataFrame {
                        slot: frame.slot,
                        operation: frame.operation.clone(),
                        mask_size: frame.mask_size,
                        heuristics: vec!["bytes".to_string()],
                    },
                    potential_types.to_owned(),
                ),
            );
        }
    }
}

static NON_PURE_OPCODES: [&str; 24] = [
    "BALANCE",
    "ORIGIN",
    "CALLER",
    "GASPRICE",
    "EXTCODESIZE",
    "EXTCODECOPY",
    "BLOCKHASH",
    "COINBASE",
    "TIMESTAMP",
    "NUMBER",
    "DIFFICULTY",
    "GASLIMIT",
    "CHAINID",
    "SELFBALANCE",
    "BASEFEE",
    "SLOAD",
    "SSTORE",
    "CREATE",
    "SELFDESTRUCT",
    "CALL",
    "CALLCODE",
    "DELEGATECALL",
    "STATICCALL",
    "CREATE2",
];

static NON_VIEW_OPCODES: [&str; 8] = [
    "SSTORE",
    "CREATE",
    "SELFDESTRUCT",
    "CALL",
    "CALLCODE",
    "DELEGATECALL",
    "STATICCALL",
    "CREATE2",
];

static MATH_OPCODES: [&str; 13] = [
    "MUL",
    "MULMOD",
    "ADDMOD",
    "SMOD",
    "MOD",
    "DIV",
    "SDIV",
    "EXP",
    "LT",
    "GT",
    "SLT",
    "SGT",
    "SIGNEXTEND",
];

static BITWISE_OPCODES: [&str; 5] = ["SHR", "SHL", "SAR", "XOR", "BYTE"];

static LOG_OPCODES: [u8; 5] = [0xA0, 0xA1, 0xA2, 0xA3, 0xA4];
