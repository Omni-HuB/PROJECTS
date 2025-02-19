module TopModule (
  input wire clk,
  input wire rst,
  input wire [31:0] address,
  output wire [31:0] data,
  output wire valid,
  input wire ready
);

  // Instantiate the L1 data cache module
  L1DCache l1_cache (
    .clk(clk),
    .rst(rst),
    .address(address),
    .data(data),
    .valid(valid)
  );

  // Instantiate the main memory module
  MainMemory main_memory (
    .clk(clk),
    .rst(rst),
    .address(address),
    .data(data),
    .valid(valid),
    .ready(ready)
  );

endmodule
