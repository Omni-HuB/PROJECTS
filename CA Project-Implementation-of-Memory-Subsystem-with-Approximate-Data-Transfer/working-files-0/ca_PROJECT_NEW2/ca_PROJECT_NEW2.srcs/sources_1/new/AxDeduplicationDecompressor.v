`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/21/2023 10:27:26 PM
// Design Name: 
// Module Name: AxDeduplicationDecompressor
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

module AxDeduplicationDecompressor(
  input wire [7:0] compressed_data,
  output wire [7:0] decompressed_data
);

  // Internal signals
  reg [7:0] base_element;
  reg [2:0] count;
  reg [7:0] decompressed_value;

  // Decompression process
  always @* begin
    base_element = compressed_data[7:0];
    count = compressed_data[15:8];

    decompressed_value = 0; // Initialize decompressed value

    // Repeat the base element according to the count
    
    for (integer i = 0; i < count; i = i + 1) begin
      decompressed_value = {decompressed_value, base_element};
    end
  end

  // Output the decompressed data
  assign decompressed_data = decompressed_value;

endmodule
