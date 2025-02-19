`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 17.10.2023 12:04:05
// Design Name: 
// Module Name: L1DCache
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module L1DCache (
    input wire clk,
    input wire rst,
    input wire [31:0] address,
    output reg [31:0] data,
    output reg valid
);

//Memory Request and Response Signals: These signals are used for communication with the main memory

// Declare memory request signals
reg [31:0] memory_request_address;
reg memory_request_valid;

// Declare memory response signals
wire memory_response_ready;
wire [31:0] memory_response_data;


// Define cache parameters
parameter CACHE_SIZE = 2048; // 2 KB
parameter CACHE_WAY = 4;
parameter CACHE_LINE_SIZE = 32; // 32 bytes

// Declare cache storage
reg [31:0] cache_data [0:CACHE_SIZE / CACHE_LINE_SIZE - 1][0:CACHE_WAY - 1];
reg [31:0] cache_tags [0:CACHE_SIZE / CACHE_LINE_SIZE - 1][0:CACHE_WAY - 1];
reg cache_valid [0:CACHE_SIZE / CACHE_LINE_SIZE - 1][0:CACHE_WAY - 1];

// Cache state
reg [31:0] cache_out_data;
reg cache_hit;

// Implement cache logic 
always @(*) begin
    // Cache hit logic
    cache_hit = 0;
    for (integer way = 0; way < CACHE_WAY; way = way + 1) begin
        if (cache_valid[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][way] &&
            cache_tags[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][way] == address[CACHE_SIZE-1:5]) begin
            cache_hit = 1;
            // Set cache_out_data to the data from the cache
            cache_out_data = cache_data[address[CACHE_SIZE-1:CACHE_LINE_SIZE]][way];
        end
    end
end

// Logic for handling cache misses and initiating main memory request
always @(posedge clk or posedge rst) begin
    if (rst) begin
        // Reset cache and other signals

        for (integer i = 0; i < CACHE_SIZE / CACHE_LINE_SIZE; i = i + 1) begin
            for (integer way = 0; way < CACHE_WAY; way = way + 1) begin
                cache_valid[i][way] = 0;
            end
        end
        // ...
    end else begin
        // Output valid signal
        valid <= ~cache_hit; // Set valid if it's a cache miss

        // Output data (4-byte word) if cache hit
        data <= cache_out_data;

        if (~cache_hit) begin

            memory_request_address = address;
            memory_request_valid = 1;

            
            // Wait for the memory response
            if (memory_response_ready) begin
                // Retrieve the data from the memory response
                cache_out_data = memory_response_data;

                // Memory request completed
                memory_request_valid = 0;
            end
        end
    end
end

endmodule



