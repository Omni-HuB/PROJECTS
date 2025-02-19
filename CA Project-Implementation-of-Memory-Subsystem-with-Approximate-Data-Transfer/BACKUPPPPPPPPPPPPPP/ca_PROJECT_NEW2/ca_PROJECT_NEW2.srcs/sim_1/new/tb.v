module Testbench;

  // Inputs
  reg clk;
  reg rst;
  reg [31:0] address;
  // Outputs
  wire [31:0] data;
  wire valid;
  wire ready;

  // Instantiate the top module
  TopModule top_module (
    .clk(clk),
    .rst(rst),
    .address(address),
    .data(data),
    .valid(valid),
    .ready(ready)
  );

  // Clock generation
  always begin
    #5 clk = ~clk;
  end

  // Test case 1: Cache hit
  initial begin
  
    $display("Test case 1: Cache hit");
    clk = 0;
    rst = 1;
    address = 32'h00010000;
    #10 rst = 0;
    #20 address = 32'h00010004;
    if (valid === 1) begin
      if (data === 32'hDEADBEEF) begin
        if (ready === 1) begin
          $display("Test case 1: Cache hit passed");
        end else begin
          $display("Test case 1: Cache hit failed (ready)");
        end
      end else begin
        $display("Test case 1: Cache hit failed (data)");
      end
    end else begin
      $display("Test case 1: Cache hit failed (valid)");
    end
  end

  // Test case 2: Cache miss
  initial begin
 
    $display("Test case 2: Cache miss");
    clk = 0;
    rst = 1;
    address = 32'h00030000;
    #10 rst = 0;
    #20 address = 32'h00040000;
    if (valid === 0) begin
      address = 32'h00030000;
      if (valid === 1) begin
        if (data === 32'hFACEFEED) begin
          if (ready === 1) begin
            $display("Test case 2: Cache miss passed");
          end else begin
            $display("Test case 2: Cache miss failed (ready)");
          end
        end else begin
          $display("Test case 2: Cache miss failed (data)");
        end
      end else begin
        $display("Test case 2: Cache miss failed (valid)");
      end
    end else begin
      $display("Test case 2: Cache miss failed (valid)");
    end
  end

  // Finish simulation
  initial begin
    #1000;
    $finish;
  end

endmodule